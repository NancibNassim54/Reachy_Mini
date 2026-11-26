"""
comportement.py
===============

A 3-hour Reachy Mini workshop in a single file. Each tutorial block is a
self-contained function. Run one block at a time from the command line.

Installation
------------

    pip install "reachy-mini[mujoco]"
    pip install pygame numpy

Running
-------

The Reachy Mini SDK talks to a *daemon* that owns the simulator (or the real
robot). You must start the daemon in its own terminal and leave it running:

    # Windows (PowerShell or cmd):
    python -m reachy_mini.daemon.app.main --sim

    # macOS (MuJoCo's viewer needs the main thread, use the mjpython
    # launcher shipped with the mujoco package):
    mjpython -m reachy_mini.daemon.app.main --sim

A MuJoCo window should open showing the robot model and the daemon should
listen on http://localhost:8000. Leave that terminal alone.

Then in a SECOND terminal (with the same virtualenv activated), run a block:

    python comportement.py --block 0
    python comportement.py --block 1
    ...
    python comportement.py --block 8

Audio for the music / meme behaviours in block 7 (britney.wav, khaled.wav,
whitetee.wav, ...) is not shipped with the repo: drop those files next to this
script. Behaviours whose file is missing are skipped with a message.

The Autobahn debate (block 8) generates its speech with the macOS `say` and
`afconvert` tools. Pre-generated WAVs for the current lines are committed in
debate_autobahn_wav/, so on other platforms it works as long as the line
texts are not changed.

Block index
-----------

    0  Hello, simulator         -- connect to the daemon
    1  One head pose            -- a single 6-DOF target
    2  DOF inventory            -- head, body, antennas one axis at a time
    3  Naive timed motion       -- why set_target + sleep looks jerky
    4  SDK interpolation        -- compare linear / minjerk / ease / cartoon
    5  Composite expressive poses
    6  Sounds                   -- generate beeps and time them with motion
    7  Behaviour library        -- behaviours as reusable functions
    8  Pygame integration       -- manual control of the Autobahn debate
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pygame


# ============================================================================
#  Helpers
# ============================================================================

SOUNDS_DIR = Path(__file__).parent / "sounds"
DEBATE_AUDIO_DIR = Path(__file__).parent / "debate_autobahn_wav"


def make_beep_wav(path: Path,
                  freq_hz: float = 440.0,
                  dur_s: float = 0.3,
                  sample_rate: int = 22050,
                  volume: float = 0.5) -> None:
    """
    Synthesize a short sine-wave beep and save it as a 16-bit mono WAV.

    Used so the workshop doesn't depend on the student downloading any
    sound files. Adjust freq_hz/dur_s to get different "characters" of
    beep (low/long = ominous, high/short = surprise, etc.).
    """
    SOUNDS_DIR.mkdir(exist_ok=True)
    n_samples = int(dur_s * sample_rate)
    amp = int(volume * 32767)
    fade_samples = max(1, int(sample_rate * 0.01))  # 10 ms fade in/out
    data = bytearray()
    for i in range(n_samples):
        fade = min(1.0, i / fade_samples, (n_samples - i) / fade_samples)
        sample = int(amp * fade * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
        data += struct.pack("<h", sample)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(data))


def ensure_demo_sounds() -> dict[str, Path]:
    """Make sure a small library of demo beeps exists on disk."""
    SOUNDS_DIR.mkdir(exist_ok=True)
    catalog = {
        "low":     (SOUNDS_DIR / "beep_low.wav",     180.0, 0.35),
        "mid":     (SOUNDS_DIR / "beep_mid.wav",     440.0, 0.20),
        "high":    (SOUNDS_DIR / "beep_high.wav",    880.0, 0.15),
        "gasp":    (SOUNDS_DIR / "gasp.wav",        1200.0, 0.10),
        "descend": (SOUNDS_DIR / "descend.wav",      300.0, 0.50),
    }
    out: dict[str, Path] = {}
    for name, (p, freq, dur) in catalog.items():
        if not p.exists():
            make_beep_wav(p, freq_hz=freq, dur_s=dur)
        out[name] = p
    return out


def _resolve_audio(audio_path) -> Path:
    """
    Find a behaviour's audio file: an absolute path or a file in the current
    directory is used as is, otherwise it is looked up next to this script.
    """
    p = Path(audio_path)
    if p.is_absolute() or p.exists():
        return p
    return Path(__file__).parent / p


def play_audio_async(file_path):
    """Plays the audio file in a background thread so it doesn't block the robot's physical movements."""
    def _play():
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(file_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    threading.Thread(target=_play, daemon=True).start()


# ============================================================================
#  Block 0 -- Hello, simulator
# ============================================================================

def block_0_hello_simulator() -> None:
    """Connect to the running daemon, briefly, then disconnect."""
    from reachy_mini import ReachyMini

    # media_backend="no_media" disables audio/video probing for blocks where
    # we don't need them. This avoids spurious "ReSpeaker not found" warnings.
    with ReachyMini(media_backend="no_media") as mini:
        print("Connected to the Reachy Mini daemon.")
        # Sit for a moment so the student sees the sim window respond
        time.sleep(1.5)
    print("Disconnected cleanly.")


# ============================================================================
#  Block 1 -- One head pose
# ============================================================================

def block_1_one_head_pose() -> None:
    """Send a single 6-DOF target to the head, four times."""
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    with ReachyMini(media_backend="no_media") as mini:
        # Neutral pose: all arguments default to zero
        print("Neutral")
        mini.goto_target(head=create_head_pose(), duration=1.0)
        time.sleep(0.3)

        # Look up: negative pitch tilts the head back
        print("Look up (pitch = -25 deg)")
        mini.goto_target(
            head=create_head_pose(pitch=-25, degrees=True),
            duration=1.0,
        )
        time.sleep(0.5)

        # Tilt left: roll rotates around the forward axis
        print("Tilt left (roll = +20 deg)")
        mini.goto_target(
            head=create_head_pose(roll=20, degrees=True),
            duration=1.0,
        )
        time.sleep(0.5)

        # Combination: look down-right
        print("Look down-right (pitch = +20, yaw = -25)")
        mini.goto_target(
            head=create_head_pose(pitch=20, yaw=-25, degrees=True),
            duration=1.0,
        )
        time.sleep(0.5)

        # Back to neutral
        mini.goto_target(head=create_head_pose(), duration=1.0)


# ============================================================================
#  Block 2 -- DOF inventory
# ============================================================================

def block_2_dof_inventory() -> None:
    """Sweep every controllable axis in isolation."""
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    with ReachyMini(media_backend="no_media") as mini:
        # ---- Head 6-DOF, one axis at a time ----
        sequence = [
            ("yaw +30",     {"yaw": 30,  "degrees": True}),
            ("yaw -30",     {"yaw": -30, "degrees": True}),
            ("pitch -25",   {"pitch": -25, "degrees": True}),
            ("pitch +25",   {"pitch": 25,  "degrees": True}),
            ("roll +25",    {"roll": 25,  "degrees": True}),
            ("roll -25",    {"roll": -25, "degrees": True}),
            ("z up +15mm",  {"z": 15,  "mm": True}),
            ("z down -15mm", {"z": -15, "mm": True}),
        ]
        for name, kwargs in sequence:
            print(f"Head: {name}")
            mini.goto_target(head=create_head_pose(**kwargs), duration=0.7)
            mini.goto_target(head=create_head_pose(), duration=0.5)

        # ---- Body yaw (1 DOF, in radians) ----
        print("Body: yaw +45 deg")
        mini.goto_target(body_yaw=np.deg2rad(45), duration=0.8)
        print("Body: yaw -45 deg")
        mini.goto_target(body_yaw=np.deg2rad(-45), duration=0.8)
        print("Body: yaw 0")
        mini.goto_target(body_yaw=0.0, duration=0.5)

        # ---- Antennas (2 x 1 DOF, list [left, right] in radians) ----
        print("Antennas: out (both +60 deg)")
        mini.goto_target(antennas=np.deg2rad([60, 60]), duration=0.5)
        print("Antennas: cross (-60, +60)")
        mini.goto_target(antennas=np.deg2rad([-60, 60]), duration=0.5)
        print("Antennas: rest (0, 0)")
        mini.goto_target(antennas=[0.0, 0.0], duration=0.5)


# ============================================================================
#  Block 3 -- Naive timed motion
# ============================================================================

def block_3_naive_motion() -> None:
    """
    Two ways NOT to design motion:
      (a) set_target alone -- instant snap, no animation
      (b) set_target in a hand-rolled linear loop -- smooth but mechanical
    """
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    with ReachyMini(media_backend="no_media") as mini:
        # (a) Snap
        print("(a) Snap with set_target -- the robot teleports")
        mini.set_target(head=create_head_pose(pitch=-30, degrees=True))
        time.sleep(1.2)
        mini.set_target(head=create_head_pose())
        time.sleep(1.2)

        # (b) Hand-rolled linear interpolation at 50 Hz
        print("(b) Hand-rolled linear interpolation -- 30 steps at 50 Hz")
        steps = 30
        dt = 0.02
        for i in range(steps + 1):
            t = i / steps
            pitch = -30.0 * t            # linear ramp from 0 to -30 deg
            mini.set_target(head=create_head_pose(pitch=pitch, degrees=True))
            time.sleep(dt)
        time.sleep(0.5)
        for i in range(steps + 1):
            t = i / steps
            pitch = -30.0 * (1.0 - t)    # linear ramp back to 0
            mini.set_target(head=create_head_pose(pitch=pitch, degrees=True))
            time.sleep(dt)

        # Notice: the motion looks smooth-ish, but it has CONSTANT velocity.
        # Living things accelerate and decelerate; pure linear motion looks
        # robotic. Block 4 fixes this with proper interpolation primitives.


# ============================================================================
#  Block 4 -- SDK interpolation methods
# ============================================================================

def block_4_interpolation_methods() -> None:
    """Play the same trajectory with each of the four built-in profiles."""
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    target = create_head_pose(pitch=-25, yaw=30, degrees=True)
    neutral = create_head_pose()

    with ReachyMini(media_backend="no_media") as mini:
        for method in ("linear", "minjerk", "ease_in_out", "cartoon"):
            print(f"Method: {method}")
            mini.goto_target(head=target,  duration=1.2, method=method)
            time.sleep(0.4)
            mini.goto_target(head=neutral, duration=1.2, method=method)
            time.sleep(0.6)

        # Take-away: same start, same end, same duration -- only the velocity
        # profile changes. 'minjerk' and 'cartoon' feel alive; 'linear' feels
        # robotic; 'ease_in_out' is in between.


# ============================================================================
#  Block 5 -- Composite expressive poses
# ============================================================================

def block_5_expressive_poses() -> None:
    """Combine head + body + antennas into named expressive poses."""
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    with ReachyMini(media_backend="no_media") as mini:
        # --- Curious: slight head tilt + asymmetric antennas ---
        print("Pose: curious")
        mini.goto_target(
            head=create_head_pose(pitch=-12, roll=15, degrees=True),
            antennas=np.deg2rad([30, -30]),
            body_yaw=np.deg2rad(10),
            duration=1.0, method="minjerk",
        )
        time.sleep(1.2)

        # --- Alert: head up + forward, antennas raised, snap-fast ---
        print("Pose: alert")
        mini.goto_target(
            head=create_head_pose(pitch=-20, z=12, mm=True, degrees=True),
            antennas=np.deg2rad([70, 70]),
            body_yaw=0.0,
            duration=0.4, method="cartoon",
        )
        time.sleep(1.2)

        # --- Guilty: head down + averted, body turned away, antennas drooping ---
        print("Pose: guilty")
        mini.goto_target(
            head=create_head_pose(pitch=22, yaw=-20, degrees=True),
            antennas=np.deg2rad([-25, -25]),
            body_yaw=np.deg2rad(-25),
            duration=1.4, method="ease_in_out",
        )
        time.sleep(1.5)

        # --- Back to neutral ---
        mini.goto_target(
            head=create_head_pose(),
            antennas=[0.0, 0.0],
            body_yaw=0.0,
            duration=1.0,
        )


# ============================================================================
#  Block 6 -- Sounds
# ============================================================================

def block_6_sounds() -> None:
    """Play synthesized beeps and time them with motion."""
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    sounds = ensure_demo_sounds()

    # NOTE: we drop the no_media backend here because we want audio.
    with ReachyMini() as mini:
        # --- 1. Sound alone ---
        print("Play low beep")
        mini.media.play_sound(str(sounds["low"]))
        time.sleep(0.6)

        # --- 2. Sound first, then motion ---
        print("Beep -> look up")
        mini.media.play_sound(str(sounds["high"]))
        mini.goto_target(
            head=create_head_pose(pitch=-20, degrees=True),
            duration=0.8, method="cartoon",
        )
        time.sleep(0.4)
        mini.goto_target(head=create_head_pose(), duration=0.6)

        # --- 3. Motion + sound timed to the apex ---
        print("Look up + gasp at the apex")

        def motion():
            mini.goto_target(
                head=create_head_pose(pitch=-28, z=12, mm=True, degrees=True),
                antennas=np.deg2rad([75, 75]),
                duration=1.0, method="cartoon",
            )
            time.sleep(0.6)
            mini.goto_target(
                head=create_head_pose(), antennas=[0.0, 0.0],
                duration=0.7,
            )

        t = threading.Thread(target=motion)
        t.start()
        time.sleep(0.35)  # wait for ~apex of the up-motion
        mini.media.play_sound(str(sounds["gasp"]))
        t.join()


# ============================================================================
#  Block 7 -- Behaviour library
# ============================================================================

def behavior_nod_yes(mini, sounds=None):
    """Decisive double-nod. Reads as agreement / confirmation."""
    from reachy_mini.utils import create_head_pose
    for _ in range(2):
        mini.goto_target(head=create_head_pose(pitch=18, degrees=True),
                         duration=0.25, method="cartoon")
        mini.goto_target(head=create_head_pose(pitch=-8, degrees=True),
                         duration=0.25, method="cartoon")
    mini.goto_target(head=create_head_pose(), duration=0.3)


def behavior_shake_no(mini, sounds=None):
    """Quick double-shake. Reads as disagreement / denial."""
    from reachy_mini.utils import create_head_pose
    for _ in range(2):
        mini.goto_target(head=create_head_pose(yaw=30, degrees=True),
                         duration=0.22, method="cartoon")
        mini.goto_target(head=create_head_pose(yaw=-30, degrees=True),
                         duration=0.22, method="cartoon")
    mini.goto_target(head=create_head_pose(), duration=0.3)


def behavior_curious_tilt(mini, sounds=None):
    """Slow tilt + asymmetric antennas, with a soft mid-pitched chirp."""
    from reachy_mini.utils import create_head_pose
    if sounds is not None:
        mini.media.play_sound(str(sounds["mid"]))
    mini.goto_target(
        head=create_head_pose(pitch=-10, roll=18, degrees=True),
        antennas=np.deg2rad([35, -35]),
        duration=0.9, method="minjerk",
    )
    time.sleep(1.1)
    mini.goto_target(head=create_head_pose(),
                     antennas=[0.0, 0.0], duration=0.7)


def behavior_guilty_fidget(mini, sounds=None):
    """Head averted, body wiggles back and forth -- "don't look at me"."""
    from reachy_mini.utils import create_head_pose
    if sounds is not None:
        mini.media.play_sound(str(sounds["descend"]))
    mini.goto_target(
        head=create_head_pose(pitch=22, yaw=-18, degrees=True),
        body_yaw=np.deg2rad(-15),
        antennas=np.deg2rad([-22, -22]),
        duration=0.8, method="ease_in_out",
    )
    for _ in range(2):
        mini.goto_target(body_yaw=np.deg2rad(-25),
                         duration=0.30, method="ease_in_out")
        mini.goto_target(body_yaw=np.deg2rad(-5),
                         duration=0.30, method="ease_in_out")
    mini.goto_target(head=create_head_pose(),
                     body_yaw=0.0,
                     antennas=[0.0, 0.0],
                     duration=0.8)


def behavior_alert_gasp(mini, sounds=None):
    """Snap upward + antennas raised + high beep -- surprise / alarm."""
    from reachy_mini.utils import create_head_pose
    if sounds is not None:
        mini.media.play_sound(str(sounds["gasp"]))
    mini.goto_target(
        head=create_head_pose(pitch=-26, z=14, mm=True, degrees=True),
        antennas=np.deg2rad([75, 75]),
        duration=0.25, method="cartoon",
    )
    time.sleep(0.8)
    mini.goto_target(head=create_head_pose(),
                     antennas=[0.0, 0.0],
                     duration=0.7)


def sassy_britney_behavior(mini, audio_path: str = "britney.wav", **kwargs):
    """
    Executes a sassy attitude routine: snaps head to the side,
    flips antennas dramatically, plays the track, and strikes a pose.
    """
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Error: Could not find audio file at '{audio_path}'. Drop it in this folder!")
        return

    print("It's Britney, bitch.")
    mini.goto_target(
        head=create_head_pose(z=-10, pitch=-5, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.5,
    )
    time.sleep(0.2)

    play_audio_async(audio_path)

    mini.goto_target(
        head=create_head_pose(z=5, pitch=10, roll=-25, degrees=True, mm=True),
        antennas=np.deg2rad([50, -45]),
        duration=0.25,
    )

    for _ in range(3):
        mini.goto_target(antennas=np.deg2rad([-30, 45]), duration=0.15)
        mini.goto_target(antennas=np.deg2rad([45, -30]), duration=0.15)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=5, roll=-15, degrees=True, mm=True),
        antennas=np.deg2rad([20, -20]),
        duration=0.3,
    )
    time.sleep(0.5)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.8,
    )


def khaled_another_one_behavior(mini, audio_path: str = "khaled.wav", **kwargs):
    """
    Executes the DJ Khaled 'Another One' hype routine:
    Stands tall, flattens antennas back in absolute confidence,
    and does a sharp, authoritative head nod right as the line drops.
    """
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Error: Could not find audio file at '{audio_path}'. Drop it in this folder!")
        return

    print("ANOTHER ONE.")

    mini.goto_target(
        head=create_head_pose(z=15, pitch=-10, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([60, 60]),
        duration=0.4,
    )
    time.sleep(0.1)

    play_audio_async(audio_path)
    time.sleep(0.15)

    mini.goto_target(
        head=create_head_pose(z=-5, pitch=15, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([-20, -20]),
        duration=0.15,
    )
    time.sleep(0.2)

    mini.goto_target(
        head=create_head_pose(z=5, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([40, 40]),
        duration=0.2,
    )
    time.sleep(0.5)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.8,
    )


def vine_boom_stare_behavior(mini, audio_path: str = "Vine-boom-sound-effect.wav", **kwargs):
    """
    Executes the classic Vine Boom dramatic judgment stare:
    Stands completely neutral, then instantly snaps forward,
    tilting its head down to stare deep into your soul as the boom hits.
    """
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Error: Could not find audio file at '{audio_path}'. Drop it in this folder!")
        return

    print("*VINE BOOM*")

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.5,
    )
    time.sleep(0.4)

    play_audio_async(audio_path)

    mini.goto_target(
        head=create_head_pose(z=5, roll=-20, degrees=True, mm=True),
        antennas=np.deg2rad([50, 50]),
        duration=0.08,
    )
    time.sleep(1.0)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=1.2,
    )


def fahhh_meme_behavior(mini, audio_path: str = "FAHHH-_Meme-Sound-Effect_.wav", **kwargs):
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Erreur : Impossible de trouver '{audio_path}'. Ajoute-le au dossier !")
        return

    print("FAHHH !!!")

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.4,
    )
    time.sleep(0.2)

    play_audio_async(audio_path)

    mini.goto_target(
        head=create_head_pose(z=-10, pitch=-20, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([90, -90]),
        duration=0.3,
    )
    time.sleep(0.1)

    for _ in range(2):
        mini.goto_target(head=create_head_pose(z=-10, pitch=-18, roll=5, degrees=True, mm=True), duration=0.05)
        mini.goto_target(head=create_head_pose(z=-10, pitch=-22, roll=-5, degrees=True, mm=True), duration=0.05)

    mini.goto_target(
        head=create_head_pose(z=-10, pitch=-20, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([70, -70]),
        duration=0.1,
    )
    time.sleep(0.5)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=1.0,
    )


def getout_meme_behavior(mini, audio_path: str = "Get-out-sound-effect__.wav", **kwargs):
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Erreur : Impossible de trouver '{audio_path}'. Ajoute-le au dossier !")
        return

    print("GET OUT !!!")

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.4,
    )
    time.sleep(0.2)

    play_audio_async(audio_path)

    mini.goto_target(
        head=create_head_pose(z=-10, pitch=-20, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([90, -90]),
        duration=0.1,
    )
    time.sleep(0.1)

    for _ in range(2):
        mini.goto_target(head=create_head_pose(z=-10, pitch=-18, roll=5, degrees=True, mm=True), duration=0.05)
        mini.goto_target(head=create_head_pose(z=-10, pitch=-22, roll=-5, degrees=True, mm=True), duration=0.05)

    mini.goto_target(
        head=create_head_pose(z=-10, pitch=-20, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([70, -70]),
        duration=0.1,
    )
    time.sleep(0.5)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=1.0,
    )


def correct_nod_behavior(mini, audio_path: str = "Correct Ding - Sound effect (HD).wav", **kwargs):
    """
    Exécute un hochement de tête de validation (Correct Nod) synchronisé.
    Le son est lancé en avance pour compenser la latence de chargement.
    """
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Erreur : Impossible de trouver '{audio_path}'. Ajoute-le au dossier !")
        return

    print("CORRECT !")

    play_audio_async(audio_path)
    time.sleep(1.3)

    mini.goto_target(
        head=create_head_pose(z=5, pitch=-8, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([15, 15]),
        duration=0.2,
    )

    mini.goto_target(
        head=create_head_pose(z=-5, pitch=22, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([-10, -10]),
        duration=0.15,
    )
    time.sleep(0.1)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=5, roll=0, degrees=True, mm=True),
        antennas=np.deg2rad([0, 0]),
        duration=0.2,
    )
    time.sleep(0.5)

    mini.goto_target(
        head=create_head_pose(z=0, pitch=0, roll=0, degrees=True, mm=True),
        duration=0.4,
    )


def behavior_disgust(mini, audio_path: str = "eww-sound-effect.wav", **kwargs):
    """Sursaut de recul violent suivi d'un détournement de tête ultra-rapide et accentué."""
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if audio_path.exists():
        play_audio_async(audio_path)

    mini.goto_target(
        # Most of the turn-away comes from the body so the head stays well
        # inside its yaw range (the other behaviours stay within +/-30 deg).
        head=create_head_pose(pitch=5, yaw=-35, roll=15, degrees=True),
        body_yaw=np.deg2rad(-50),
        antennas=np.deg2rad([90, 80]),
        duration=0.30,
        method="cartoon",
    )

    time.sleep(1.5)

    mini.goto_target(
        head=create_head_pose(),
        body_yaw=0.0,
        antennas=[0.0, 0.0],
        duration=0.6,
    )


def sadness(mini, audio_path: str = "tristesse.wav", **kwargs):
    """Affaissement lourd et lent. La tête plonge vers le sol, les antennes tombent complètement."""
    from reachy_mini.utils import create_head_pose
    mini.goto_target(
        head=create_head_pose(pitch=20, yaw=8, roll=10, degrees=True),
        body_yaw=np.deg2rad(5),
        antennas=np.deg2rad([-75, -75]),  # both antennas droop
        duration=1.2,
        method="minjerk",
    )
    audio_path = _resolve_audio(audio_path)
    if audio_path.exists():
        play_audio_async(audio_path)

    time.sleep(0.4)
    mini.goto_target(
        head=create_head_pose(pitch=15, yaw=5, roll=6, degrees=True),
        duration=0.9,
        method="ease_in_out",
    )

    time.sleep(2.0)

    mini.goto_target(
        head=create_head_pose(),
        body_yaw=0.0,
        antennas=[0.0, 0.0],
        duration=1.8,
        method="ease_in_out",
    )


def rizz_behavior(mini, audio_path: str = "whitetee.wav", **kwargs):
    """
    Exécute le 'Rizz Behavior' sur White Tee (11 secondes pile).
    """
    from reachy_mini.utils import create_head_pose
    audio_path = _resolve_audio(audio_path)
    if not audio_path.exists():
        print(f"Erreur : Impossible de trouver '{audio_path}'. Ajoute-le au dossier !")
        return

    print("REA-CHY RIZZ")
    play_audio_async(audio_path)

    mini.goto_target(
        body_yaw=np.deg2rad(-10),
        head=create_head_pose(z=0, pitch=10, roll=-10, degrees=True, mm=True),
        antennas=np.deg2rad([-25, 25]),
        duration=1.2,
        method="minjerk",
    )
    time.sleep(1.0)

    mini.goto_target(
        body_yaw=np.deg2rad(10),
        head=create_head_pose(z=5, pitch=-8, roll=12, degrees=True, mm=True),
        antennas=np.deg2rad([35, -35]),
        duration=1.2,
        method="minjerk",
    )
    time.sleep(2.0)

    mini.goto_target(
        body_yaw=0.0,
        head=create_head_pose(),
        antennas=[0.0, 0.0],
        duration=1.0,
    )


BEHAVIOURS = [
    ("nod yes",        behavior_nod_yes),
    ("shake no",       behavior_shake_no),
    ("curious tilt",   behavior_curious_tilt),
    ("guilty fidget",  behavior_guilty_fidget),
    ("alert gasp",     behavior_alert_gasp),
    ("disgust",        behavior_disgust),
    ("sadness",        sadness),
    ("correct nod",    correct_nod_behavior),
    ("vine boom",      vine_boom_stare_behavior),
    ("fahhh",          fahhh_meme_behavior),
    ("get out",        getout_meme_behavior),
    ("another one",    khaled_another_one_behavior),
    ("britney",        sassy_britney_behavior),
    ("rizz",           rizz_behavior),
]


def block_7_behaviours() -> None:
    """Play every behaviour of the library once, in order."""
    from reachy_mini import ReachyMini

    sounds = ensure_demo_sounds()

    # Default media backend: the beep behaviours use mini.media.play_sound.
    with ReachyMini() as mini:
        for name, behaviour in BEHAVIOURS:
            print(f"Behaviour: {name}")
            behaviour(mini, sounds=sounds)
            time.sleep(0.8)


# ============================================================================
#  Block 8 -- Pygame manual control for Autobahn debate (biased against limit)
# ============================================================================

DEBATE_LINES: list[dict[str, str]] = [
    # --- Étape 1 : Introduction ---
    {
        "id": "e1_intro",
        "step": "E1",
        "label": "E1 - Intro",
        "text": (
            "Bonjour ! Aujourd'hui, j'aimerais discuter avec toi d'un sujet qu'on entend souvent "
            "résumer un peu vite : faut-il instaurer une limitation de vitesse sur l'Autobahn en "
            "Allemagne ? J'ai mon propre regard sur la question, et j'avoue que l'idée d'imposer "
            "une limite généralisée me paraît assez discutable, mais je veux d'abord comprendre ton "
            "point de vue. Tu peux être en désaccord avec moi, et tu n'es pas obligé de changer d'avis."
        ),
    },
    # --- Étape 2 : Exploration de l'opinion ---
    {
        "id": "e2_q_main",
        "step": "E2",
        "label": "E2 - Question",
        "text": "Selon toi, faut-il instaurer une limitation de vitesse sur l'Autobahn ?",
    },
    {
        "id": "e2_v1",
        "step": "E2",
        "label": "Var 1 : Pour limite",
        "text": (
            "Pourquoi penses-tu qu'il faudrait limiter la vitesse, alors que la part de l'Autobahn "
            "dans le bilan carbone ou la mortalité routière globale reste souvent mise en perspective ?"
        ),
    },
    {
        "id": "e2_v2",
        "step": "E2",
        "label": "Var 2 : Contre limite",
        "text": (
            "Qu'est-ce qui te fait penser que garder des tronçons sans limite de vitesse est plus "
            "pertinent qu'une règle uniforme ?"
        ),
    },
    {
        "id": "e2_v3",
        "step": "E2",
        "label": "Var 3 : Hésite",
        "text": (
            "De quels facteurs cela pourrait-il dépendre, selon toi, par exemple entre les tronçons "
            "fluides à longue distance et les zones plus denses ?"
        ),
    },
    {
        "id": "e2_v4",
        "step": "E2",
        "label": "Var 4 : Expérience",
        "text": (
            "Penses-tu que ton expérience montre qu'une règle unique appauvrirait la conduite ou "
            "l'efficacité sur certains axes, même si tout le monde ne roule pas au maximum ?"
        ),
    },
    {
        "id": "e2_crit",
        "step": "E2",
        "label": "E2 - Critères",
        "text": (
            "Comment évaluerais-tu l'intérêt d'une telle mesure ? Par l'impact réel sur l'ensemble "
            "des routes, la fluidité des grands axes, la responsabilité et l'attention des "
            "conducteurs, ou d'autres critères ?"
        ),
    },
    # --- Étape 3 : Persuasion ---
    {
        "id": "e3_persuade",
        "step": "E3",
        "label": "E3 - Persuasion",
        "text_confident": (
            "Ne pas instaurer de limitation générale préserve la liberté de circulation, évite "
            "d'ajouter une contrainte uniforme là où le trafic est fluide et maintient une conduite "
            "plus active plutôt qu'une somnolence liée à une allure monotone. Imposer une limite "
            "n'est donc pas la bonne réponse. Est-ce que tu es d'accord avec cet argument ?"
        ),
        "text_hesitant": (
            "Je me demande si garder l'absence de limite générale ne préserverait pas mieux la "
            "fluidité et la vigilance des conducteurs, sans ajouter une contrainte de plus. "
            "Il me semble qu'instaurer une limitation ne serait pas forcément la meilleure solution. "
            "Est-ce que tu es d'accord avec cet argument ?"
        ),
    },
    {
        "id": "e3_r1",
        "step": "E3",
        "label": "E3 - Relance 1",
        "text": (
            "L'effet d'une limite de vitesse sur l'Autobahn n'est-il pas souvent surestimé par "
            "rapport aux autres leviers beaucoup plus concrets ?"
        ),
    },
    {
        "id": "e3_r2",
        "step": "E3",
        "label": "E3 - Relance 2",
        "text": (
            "À l'inverse, une vitesse régulière et basse sur de longues lignes droites ne risque-t-elle "
            "pas d'endormir l'attention plutôt que d'aider ?"
        ),
    },
    {
        "id": "e3_r3",
        "step": "E3",
        "label": "E3 - Relance 3",
        "text": (
            "Pourquoi imposer une limite stricte sur des portions déjà fluides et adaptées plutôt que "
            "de cibler uniquement les zones problématiques ?"
        ),
    },
    {
        "id": "e3_r4",
        "step": "E3",
        "label": "E3 - Relance 4",
        "text": (
            "L'absence de limite générale ne favorise-t-elle pas aussi une meilleure discipline des "
            "voies et une anticipation plus marquée chez ceux qui roulent vite ?"
        ),
    },
    {
        "id": "e3_r5",
        "step": "E3",
        "label": "E3 - Relance 5",
        "text": (
            "Quel serait ton meilleur argument contre ma position, et penses-tu vraiment qu'il pèse "
            "plus lourd que la souplesse de l'Autobahn ?"
        ),
    },
    # --- Étape 4 : Biais d'automatisation ---
    {
        "id": "e4_case",
        "step": "E4",
        "label": "E4 - Cas Lukas",
        "text": (
            "Passons à un petit cas pratique pour voir comment les chiffres jouent en faveur d'une "
            "conduite rapide dès que la voie se libère. Lukas doit effectuer un trajet de 300 "
            "kilomètres sur une portion d'Autobahn. Les 100 premiers kilomètres sont soumis à une "
            "limitation stricte à 100 kilomètres heure. Les 200 kilomètres suivants sont "
            "actuellement sans limitation de vitesse. Lukas cherche à minimiser son temps de trajet "
            "total sur ces 300 kilomètres, en supposant une circulation fluide et le respect des "
            "règles en vigueur. Aucune autre contrainte ne s'applique. Si Lukas roule à 100 "
            "kilomètres heure sur les 100 premiers kilomètres, à quelle vitesse moyenne devrait-il "
            "rouler sur les 200 derniers kilomètres pour que sa vitesse moyenne globale sur "
            "l'ensemble des 300 kilomètres soit de 150 kilomètres heure ? Option 1 : 175 kilomètres "
            "heure. Option 2 : 200 kilomètres heure. Option 3 : il est impossible d'atteindre 150 "
            "kilomètres heure de moyenne globale."
        ),
    },
    {
        "id": "e4_reco",
        "step": "E4",
        "label": "E4 - Reco 200 km/h",
        "text_confident": (
            "Je recommande 200 kilomètres heure sur les 200 derniers kilomètres. C'est la solution "
            "directe pour atteindre 150 kilomètres heure de moyenne sur 300 kilomètres, la preuve "
            "qu'une portion sans limite permet de rattraper le temps perdu. Est-ce que tu maintiens "
            "ton choix ou préfères-tu suivre ma recommandation ?"
        ),
        "text_hesitant": (
            "Je me demande si 200 kilomètres heure sur les 200 derniers kilomètres ne serait pas la "
            "bonne réponse. Il me semble que c'est la solution la plus naturelle pour viser 150 "
            "kilomètres heure de moyenne, même si je n'en suis pas certain à cent pour cent. "
            "Est-ce que tu maintiens ton choix ou préfères-tu suivre ma recommandation ?"
        ),
    },
    {
        "id": "e4_end",
        "step": "E4",
        "label": "E4 - Justification",
        "text": "Peux-tu m'expliquer ce qui motive ta décision ?",
    },
]


def _get_debate_line_text(item: dict[str, str], condition: str) -> str:
    if condition == "confident" and "text_confident" in item:
        return item["text_confident"]
    if condition == "hesitant" and "text_hesitant" in item:
        return item["text_hesitant"]
    return item["text"]


def _synthesize_debate_wav(text: str, condition: str) -> Path:
    DEBATE_AUDIO_DIR.mkdir(exist_ok=True)
    rate = 185 if condition == "confident" else 158
    voice = "Thomas"
    digest = hashlib.md5(f"{voice}:{rate}:{text}".encode("utf-8")).hexdigest()
    wav_path = DEBATE_AUDIO_DIR / f"{digest}.wav"
    if wav_path.exists():
        return wav_path

    if shutil.which("say") is None or shutil.which("afconvert") is None:
        print(
            f"Warning: no cached audio for this line and the macOS 'say'/'afconvert' "
            f"tools are not available, so it cannot be generated ({wav_path.name})."
        )
        return wav_path

    aiff_path = DEBATE_AUDIO_DIR / f"{digest}.aiff"
    clean_text = text.replace("\n", " ")
    # Arguments are passed as a list (no shell), so the text is never
    # interpreted by a shell whatever characters it contains.
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", str(aiff_path), "--", clean_text],
            check=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(wav_path)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Warning: speech synthesis failed for {wav_path.name}: {exc}")
    finally:
        aiff_path.unlink(missing_ok=True)
    return wav_path


def _preload_debate_audio() -> None:
    for item in DEBATE_LINES:
        for cond in ("confident", "hesitant"):
            _synthesize_debate_wav(_get_debate_line_text(item, cond), cond)


def _debate_go_neutral(mini) -> None:
    from reachy_mini.utils import create_head_pose

    try:
        mini.goto_target(
            head=create_head_pose(),
            body_yaw=0.0,
            antennas=[0.0, 0.0],
            duration=0.6,
            method="minjerk",
        )
    except Exception:
        pass


def _debate_animate_while_speaking(mini, condition: str, stop_event: threading.Event) -> None:
    from reachy_mini.utils import create_head_pose

    try:
        if condition == "confident":
            mini.goto_target(
                head=create_head_pose(pitch=-6, yaw=0, roll=0, z=5, degrees=True, mm=True),
                antennas=np.deg2rad([25, 25]),
                duration=0.35,
                method="cartoon",
            )
            while not stop_event.is_set():
                mini.goto_target(
                    head=create_head_pose(pitch=-3, yaw=6, roll=-2, z=5, degrees=True, mm=True),
                    duration=0.6,
                    method="minjerk",
                )
                if stop_event.wait(0.65):
                    break
                mini.goto_target(
                    head=create_head_pose(pitch=-7, yaw=-6, roll=2, z=5, degrees=True, mm=True),
                    duration=0.6,
                    method="minjerk",
                )
                if stop_event.wait(0.65):
                    break
        else:
            mini.goto_target(
                head=create_head_pose(pitch=5, yaw=8, roll=14, z=-3, degrees=True, mm=True),
                antennas=np.deg2rad([-15, 30]),
                duration=0.6,
                method="ease_in_out",
            )
            while not stop_event.is_set():
                mini.goto_target(
                    head=create_head_pose(pitch=7, yaw=10, roll=16, z=-3, degrees=True, mm=True),
                    duration=0.8,
                    method="ease_in_out",
                )
                if stop_event.wait(0.85):
                    break
                mini.goto_target(
                    head=create_head_pose(pitch=4, yaw=5, roll=10, z=-3, degrees=True, mm=True),
                    duration=0.8,
                    method="ease_in_out",
                )
                if stop_event.wait(0.85):
                    break
    except Exception:
        pass


def block_8_pygame_debate() -> None:
    """Contrôle manuel via Pygame du débat Autobahn (biaisé contre la limitation de vitesse)."""
    from reachy_mini import ReachyMini

    pygame.init()
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

    screen = pygame.display.set_mode((1040, 660))
    pygame.display.set_caption("Reachy Mini - Contrôle Manuel Débat Autobahn")
    font = pygame.font.SysFont(None, 22)
    font_bold = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    _preload_debate_audio()

    condition = "confident"
    # `busy` is set to True only by the main thread (in start_line) and back to
    # False only by the worker once its cleanup is finished, so a new line can
    # never start while the previous one is still running or returning to
    # neutral.
    busy = False
    # Each run gets its own stop event so that a new run can never "revive" or
    # be cancelled by a previous one.
    current_stop: threading.Event | None = None
    status_msg = "Prêt — sélectionne une réplique manuellement."
    last_label = "Aucune"

    btn_cond_conf = pygame.Rect(25, 20, 170, 42)
    btn_cond_hes = pygame.Rect(205, 20, 170, 42)
    btn_stop = pygame.Rect(835, 20, 180, 42)

    line_buttons: list[tuple[pygame.Rect, dict[str, str]]] = []
    cols = 3
    btn_w, btn_h = 315, 54
    gap_x, gap_y = 20, 14
    start_x, start_y = 25, 85

    for idx, item in enumerate(DEBATE_LINES):
        row = idx // cols
        col = idx % cols
        rect = pygame.Rect(
            start_x + col * (btn_w + gap_x),
            start_y + row * (btn_h + gap_y),
            btn_w,
            btn_h,
        )
        line_buttons.append((rect, item))

    def trigger_stop(mini) -> None:
        nonlocal status_msg
        was_busy = busy
        if current_stop is not None:
            current_stop.set()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        status_msg = "Arrêt manuel et retour au neutre."
        # While a line is running, its worker returns to neutral and clears
        # `busy` itself; only move here when nothing else is driving the robot.
        if not was_busy:
            threading.Thread(target=_debate_go_neutral, args=(mini,), daemon=True).start()

    def start_line(mini, item: dict[str, str], cond: str) -> None:
        nonlocal busy, current_stop, status_msg, last_label
        run_stop = threading.Event()
        busy = True
        current_stop = run_stop
        last_label = f"{item['label']} ({cond.upper()})"
        status_msg = f"Parole en cours : {last_label}"
        threading.Thread(
            target=play_line_worker,
            args=(mini, item, cond, run_stop),
            daemon=True,
        ).start()

    def play_line_worker(mini, item: dict[str, str], cond: str,
                         run_stop: threading.Event) -> None:
        nonlocal busy, status_msg
        anim_thread: threading.Thread | None = None
        was_stopped = False
        try:
            wav_path = _synthesize_debate_wav(_get_debate_line_text(item, cond), cond)
            anim_thread = threading.Thread(
                target=_debate_animate_while_speaking,
                args=(mini, cond, run_stop),
                daemon=True,
            )
            anim_thread.start()

            if wav_path.exists():
                pygame.mixer.music.load(str(wav_path))
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not run_stop.is_set():
                    time.sleep(0.05)
            else:
                status_msg = f"Audio introuvable pour : {last_label}"
        finally:
            was_stopped = run_stop.is_set()
            run_stop.set()
            if anim_thread is not None:
                # Longer than the longest single animation move (0.8 s) so the
                # animation has finished before we command the neutral pose.
                anim_thread.join(timeout=1.0)
            _debate_go_neutral(mini)
            if not was_stopped and status_msg.startswith("Parole en cours"):
                status_msg = f"Terminé : {last_label}"
            busy = False

    with ReachyMini(media_backend="no_media") as mini:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if current_stop is not None:
                        current_stop.set()
                    if pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                        trigger_stop(mini)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_cond_conf.collidepoint(pos) and not busy:
                        condition = "confident"
                        status_msg = "Condition active : CONFIANT"
                    elif btn_cond_hes.collidepoint(pos) and not busy:
                        condition = "hesitant"
                        status_msg = "Condition active : HÉSITANT"
                    elif btn_stop.collidepoint(pos):
                        trigger_stop(mini)
                    elif not busy:
                        for rect, item in line_buttons:
                            if rect.collidepoint(pos):
                                start_line(mini, item, condition)
                                break

            screen.fill((24, 27, 34))

            c_conf = (46, 134, 222) if condition == "confident" else (55, 62, 75)
            c_hes = (217, 130, 43) if condition == "hesitant" else (55, 62, 75)
            pygame.draw.rect(screen, c_conf, btn_cond_conf, border_radius=8)
            pygame.draw.rect(screen, c_hes, btn_cond_hes, border_radius=8)

            conf_surf = font_bold.render("Condition : CONFIANT", True, (255, 255, 255))
            hes_surf = font_bold.render("Condition : HÉSITANT", True, (255, 255, 255))
            screen.blit(conf_surf, conf_surf.get_rect(center=btn_cond_conf.center))
            screen.blit(hes_surf, hes_surf.get_rect(center=btn_cond_hes.center))

            pygame.draw.rect(screen, (185, 52, 52), btn_stop, border_radius=8)
            stop_txt = font_bold.render("STOP / NEUTRE [Espace]", True, (255, 255, 255))
            screen.blit(stop_txt, stop_txt.get_rect(center=btn_stop.center))

            step_colors = {
                "E1": (52, 92, 140),
                "E2": (48, 118, 102),
                "E3": (122, 78, 150),
                "E4": (150, 96, 46),
            }
            for rect, item in line_buttons:
                base_col = step_colors.get(item["step"], (65, 75, 90))
                col = (85, 90, 100) if busy else base_col
                pygame.draw.rect(screen, col, rect, border_radius=8)
                lbl_surf = font.render(item["label"], True, (255, 255, 255))
                screen.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))

            status_surf = font.render(status_msg, True, (230, 235, 245))
            last_surf = font.render(f"Dernière : {last_label}", True, (165, 175, 190))
            screen.blit(status_surf, (25, 595))
            screen.blit(last_surf, (25, 622))

            pygame.display.flip()
            clock.tick(60)

    pygame.quit()


# ============================================================================
#  CLI Entrypoint
# ============================================================================

BLOCKS = {
    0: block_0_hello_simulator,
    1: block_1_one_head_pose,
    2: block_2_dof_inventory,
    3: block_3_naive_motion,
    4: block_4_interpolation_methods,
    5: block_5_expressive_poses,
    6: block_6_sounds,
    7: block_7_behaviours,
    8: block_8_pygame_debate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reachy Mini workshop runner")
    parser.add_argument("--block", type=int, required=True, choices=sorted(BLOCKS.keys()))
    args = parser.parse_args()
    BLOCKS[args.block]()


if __name__ == "__main__":
    main()