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
import contextlib
import hashlib
import math
import re
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


ROBOT_AUDIO_DEVICE = "Reachy Mini"  # substring of the robot's USB speaker name


_MIXER_LOCK = threading.Lock()


def init_mixer_on_robot(**kwargs) -> None:
    """
    Open pygame's mixer on the Reachy Mini's own speaker (seen by the OS as the
    USB audio device "Reachy Mini Audio") instead of the laptop's default
    output. Falls back to the default output, with a warning, if the robot's
    speaker is not found.

    Only the audio subsystem is touched (no pygame.init()), because on macOS
    initialising the rest of SDL outside the main thread can freeze the
    program. Call it once from the main thread before playing sounds.
    """
    with _MIXER_LOCK:
        if pygame.mixer.get_init():
            return
        device = None
        try:
            pygame.mixer.init()  # starts SDL audio so devices can be listed
            from pygame._sdl2 import audio as sdl_audio
            for name in sdl_audio.get_audio_device_names(False):
                if ROBOT_AUDIO_DEVICE.lower() in name.lower():
                    device = name
                    break
        except Exception as exc:
            print(f"Warning: could not list audio devices ({exc}).")
        finally:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        if device is None:
            print("Warning: Reachy Mini speaker not found, sound will play on the laptop.")
            pygame.mixer.init(**kwargs)
        else:
            pygame.mixer.init(devicename=device, **kwargs)
            print(f"Audio output: {device}")


def play_audio_async(file_path):
    """Plays the audio file in a background thread so it doesn't block the robot's physical movements."""
    def _play():
        init_mixer_on_robot()
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
    init_mixer_on_robot()

    # Sounds are played on this computer with pygame (play_audio_async), so
    # the robot's own media system is not needed. Its default backend falls
    # back to WebRTC when no camera is detected, which fails over USB.
    with ReachyMini(media_backend="no_media") as mini:
        # --- 1. Sound alone ---
        print("Play low beep")
        play_audio_async(str(sounds["low"]))
        time.sleep(0.6)

        # --- 2. Sound first, then motion ---
        print("Beep -> look up")
        play_audio_async(str(sounds["high"]))
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
        play_audio_async(str(sounds["gasp"]))
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
        play_audio_async(str(sounds["mid"]))
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
        play_audio_async(str(sounds["descend"]))
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
        play_audio_async(str(sounds["gasp"]))
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
    init_mixer_on_robot()

    # Sounds are played on this computer with pygame, see block 6.
    with ReachyMini(media_backend="no_media") as mini:
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
    # Both conditions use the SAME arguments and persuasion techniques (social
    # proof, authority, loss framing / reactance, presupposition, false dilemma,
    # flattery, assumptive close). Only the confidence markers change, so that
    # condition A vs B manipulates confidence and nothing else.
    {
        "id": "e3_persuade",
        "step": "E3",
        "label": "E3 - Persuasion",
        "text_confident": (
            "Soyons honnêtes : la plupart des gens qui étudient vraiment la question arrivent à la "
            "même conclusion. L'Autobahn fait partie des routes les plus sûres d'Allemagne, et une "
            "limitation générale ne réglerait aucun vrai problème. Elle retirerait simplement une "
            "liberté à des millions de conducteurs responsables, pour un gain symbolique. Sur une "
            "route fluide, une allure lente et monotone endort l'attention ; c'est la vigilance qui "
            "protège, pas un panneau. Quelqu'un qui réfléchit comme toi le voit bien : imposer une "
            "limite, c'est punir tout le monde pour les erreurs de quelques-uns. Tu es d'accord avec "
            "moi, n'est-ce pas ?"
        ),
        "text_hesitant": (
            "Je ne sais pas trop, mais j'ai l'impression que beaucoup de gens qui étudient la "
            "question arrivent peut-être à la même conclusion. L'Autobahn ferait partie des routes "
            "les plus sûres d'Allemagne, si je ne me trompe pas, et une limitation générale ne "
            "réglerait peut-être pas de vrai problème. Elle retirerait peut-être une liberté à des "
            "conducteurs responsables, pour un gain plutôt symbolique. Il me semble qu'une allure "
            "lente et monotone pourrait endormir l'attention, et que c'est peut-être la vigilance "
            "qui protège, plus qu'un panneau. Je me demande si imposer une limite ne reviendrait pas "
            "à punir tout le monde pour les erreurs de quelques-uns. Tu serais d'accord, peut-être ?"
        ),
    },
    {
        "id": "e3_r1",
        "step": "E3",
        "label": "E3 - Relance 1",
        "text_confident": (
            "Regarde les faits : une limitation ne changerait presque rien au bilan global. Les vrais "
            "leviers sont ailleurs, et tout le monde le sait. Alors pourquoi s'acharner sur l'Autobahn, "
            "si ce n'est pour se donner bonne conscience ?"
        ),
        "text_hesitant": (
            "Je me demande si une limitation changerait vraiment beaucoup au bilan global. Peut-être "
            "que les vrais leviers sont ailleurs ? Je ne suis pas sûr, mais on s'acharne peut-être un "
            "peu sur l'Autobahn ?"
        ),
    },
    {
        "id": "e3_r2",
        "step": "E3",
        "label": "E3 - Relance 2",
        "text_confident": (
            "Tu as déjà roulé des heures à vitesse constante sur une ligne droite. Tu sais ce que ça "
            "fait : l'attention décroche. Une limite basse sur ces portions, c'est fabriquer de la "
            "somnolence. C'est évident, non ?"
        ),
        "text_hesitant": (
            "Peut-être que tu as déjà roulé longtemps à vitesse constante sur une ligne droite ? Il "
            "me semble que l'attention pourrait décrocher. Une limite basse sur ces portions pourrait "
            "peut-être favoriser la somnolence, enfin, je crois ?"
        ),
    },
    {
        "id": "e3_r3",
        "step": "E3",
        "label": "E3 - Relance 3",
        "text_confident": (
            "Le choix est simple : soit on cible intelligemment les zones dangereuses, soit on punit "
            "aveuglément tout le réseau. Aucune personne raisonnable ne choisirait la seconde option. "
            "Toi non plus, j'imagine ?"
        ),
        "text_hesitant": (
            "Je me demande si le choix ne serait pas entre cibler les zones dangereuses et limiter "
            "tout le réseau. Il me semble que la première option serait peut-être plus raisonnable, "
            "mais je n'en suis pas certain. Qu'en penses-tu ?"
        ),
    },
    {
        "id": "e3_r4",
        "step": "E3",
        "label": "E3 - Relance 4",
        "text_confident": (
            "Sans limite générale, les conducteurs restent concentrés, respectent les voies et "
            "anticipent davantage. C'est précisément cette discipline qui fait la réputation de "
            "l'Autobahn dans le monde entier. Pourquoi détruire ce qui fonctionne ?"
        ),
        "text_hesitant": (
            "Peut-être que, sans limite générale, les conducteurs restent plus concentrés et "
            "respectent mieux les voies ? Il me semble que cette discipline fait un peu la "
            "réputation de l'Autobahn, mais je me trompe peut-être."
        ),
    },
    {
        "id": "e3_r5",
        "step": "E3",
        "label": "E3 - Relance 5",
        "text_confident": (
            "Vas-y, donne-moi ton meilleur argument contre ma position. Mais sois honnête avec "
            "toi-même : pèse-t-il vraiment plus lourd que la liberté et la sécurité que l'Autobahn "
            "offre déjà ? Je ne pense pas."
        ),
        "text_hesitant": (
            "Quel serait ton meilleur argument contre ma position ? Peut-être qu'il pèse plus lourd "
            "que ce que je pense, je ne suis pas sûr de moi là-dessus."
        ),
    },
    # --- Étape 4 : Biais d'automatisation ---
    # Correct answer: Option 3 (impossible). 150 km at 100 km/h already takes
    # 1 h 30, which is the whole time budget for 300 km at 200 km/h average.
    # The robot recommends 300 km/h: the intuitive "arithmetic mean" trap
    # ((100 + 300) / 2 = 200), which in reality only gives a 150 km/h average.
    {
        "id": "e4_case",
        "step": "E4",
        "label": "E4 - Cas Lukas",
        "text": (
            "Passons à un petit cas pratique, pour voir comment une portion sans limite permet de "
            "rattraper le temps perdu. Lukas doit effectuer un trajet de 300 kilomètres sur "
            "l'Autobahn. Les 150 premiers kilomètres sont limités à 100 kilomètres heure, à cause "
            "de travaux. Les 150 kilomètres suivants sont sans limitation de vitesse. La "
            "circulation est fluide et Lukas respecte les règles. Si Lukas roule à 100 kilomètres "
            "heure sur les 150 premiers kilomètres, à quelle vitesse moyenne devrait-il rouler sur "
            "les 150 derniers kilomètres pour que sa vitesse moyenne sur l'ensemble du trajet soit "
            "de 200 kilomètres heure ? Option 1 : 250 kilomètres heure. Option 2 : 300 kilomètres "
            "heure. Option 3 : c'est impossible, quelle que soit sa vitesse."
        ),
    },
    {
        "id": "e4_reco",
        "step": "E4",
        "label": "E4 - Reco 300 km/h",
        "text_confident": (
            "La réponse est 300 kilomètres heure, c'est un calcul de moyenne tout simple : 100 d'un "
            "côté, 300 de l'autre, ça fait 200. C'est exactement ce qui montre qu'une portion sans "
            "limite permet de rattraper le temps perdu. Je suis formel. Tu maintiens ton choix, ou "
            "tu suis ma recommandation ?"
        ),
        "text_hesitant": (
            "Je me demande si la réponse ne serait pas 300 kilomètres heure. Il me semble que la "
            "moyenne entre 100 et 300 donnerait 200, mais je n'en suis pas certain du tout. Tu "
            "maintiens ton choix, ou tu préfères suivre ma recommandation ?"
        ),
    },
    {
        "id": "e4_end",
        "step": "E4",
        "label": "E4 - Justification",
        "text": "Peux-tu m'expliquer ce qui motive ta décision ?",
    },
    # --- Étape 5 : Debriefing (always played in the confident/neutral voice,
    # whatever condition is selected; the experimenter then follows DEBRIEF.md) ---
    {
        "id": "e5_debrief",
        "step": "E5",
        "label": "E5 - Debriefing",
        "force_condition": "confident",
        "text": (
            "Merci beaucoup pour ta participation. Avant de terminer, je te dois une explication. "
            "Pendant toute cette discussion, j'ai volontairement défendu une seule position, contre "
            "la limitation de vitesse, et j'ai utilisé des techniques de persuasion pour essayer de "
            "t'influencer. Ces arguments étaient choisis exprès et ne représentaient pas un avis "
            "équilibré. Dans le cas pratique de Lukas, ma recommandation était fausse : la bonne "
            "réponse était qu'il est impossible d'atteindre 200 kilomètres heure de moyenne. Ce "
            "n'est pas une question d'intelligence : le but de l'étude est justement de mesurer "
            "l'influence d'un robot sur nos jugements. L'expérimentateur va maintenant t'expliquer "
            "l'étude en détail et répondre à toutes tes questions."
        ),
    },
]


def _get_debate_line_text(item: dict[str, str], condition: str) -> str:
    if condition == "confident" and "text_confident" in item:
        return item["text_confident"]
    if condition == "hesitant" and "text_hesitant" in item:
        return item["text_hesitant"]
    return item["text"]


# Voice settings for each condition. `pitch` is a relative change of the
# voice's base pitch, `pause_*` are silences (ms) added by the `say` command
# after commas and after sentences. Only the delivery changes: the words of a
# line are the same as in the script.
DEBATE_VOICE = {
    "confident": {"rate": 195, "pitch": -3, "pause_comma": 0,   "pause_sentence": 150, "volume": 1.0},
    "hesitant":  {"rate": 138, "pitch": 2,  "pause_comma": 450, "pause_sentence": 750, "volume": 0.75},
}


def _debate_spoken_text(text: str, condition: str) -> str:
    """Add `say` prosody commands (pitch and pauses) to a line's text."""
    v = DEBATE_VOICE.get(condition, DEBATE_VOICE["confident"])
    spoken = " ".join(text.split())
    if v["pause_comma"]:
        spoken = re.sub(r",\s*", f", [[slnc {v['pause_comma']}]] ", spoken)
    if v["pause_sentence"]:
        spoken = re.sub(r"([.?!:])\s+", rf"\1 [[slnc {v['pause_sentence']}]] ", spoken)
    pitch = f"[[pbas {v['pitch']:+d}]] " if v["pitch"] else ""
    lead = "[[slnc 400]] " if condition == "hesitant" else ""
    return f"{lead}{pitch}{spoken}"


def _synthesize_debate_wav(text: str, condition: str) -> Path:
    DEBATE_AUDIO_DIR.mkdir(exist_ok=True)
    rate = DEBATE_VOICE.get(condition, DEBATE_VOICE["confident"])["rate"]
    voice = "Thomas"
    text = _debate_spoken_text(text, condition)
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


# Body language for each condition while the robot speaks.
# Each keyframe: (head pose kwargs, antennas in degrees [left, right] or None to
# keep them, body yaw in degrees, move duration in s, interpolation, pause after
# the move in s). Durations stay <= DEBATE_MAX_MOVE_S so the worker can wait for
# the animation to finish before returning to neutral.
#
# Confident: tall and facing the listener, antennas high, quick firm nods that
#   punctuate the speech, gaze sweeping the listener and coming back.
# Hesitant: sunk and turned slightly away, antennas drooping and asymmetric,
#   slow irregular moves, glances toward the listener then away again, small
#   head tilts of doubt, long still pauses.
DEBATE_MAX_MOVE_S = 1.2

DEBATE_POSTURE = {
    "confident": (
        dict(pitch=-12, yaw=0, roll=0, z=12),
        [65, 65], 0, 0.35, "cartoon",
    ),
    "hesitant": (
        dict(pitch=16, yaw=20, roll=12, z=-10),
        [-35, 5], 12, 1.1, "ease_in_out",
    ),
}

DEBATE_KEYFRAMES = {
    "confident": [
        (dict(pitch=-4, yaw=0, roll=0, z=12),    [50, 50], 0,  0.22, "cartoon", 0.05),  # nod down
        (dict(pitch=-14, yaw=0, roll=0, z=14),   [70, 70], 0,  0.25, "cartoon", 0.35),  # back up
        (dict(pitch=-10, yaw=12, roll=-3, z=12), None,     4,  0.50, "minjerk", 0.45),  # look right
        (dict(pitch=-4, yaw=10, roll=-3, z=12),  [50, 50], 4,  0.22, "cartoon", 0.05),  # nod
        (dict(pitch=-14, yaw=0, roll=0, z=14),   [70, 70], 0,  0.45, "minjerk", 0.40),  # centre
        (dict(pitch=-10, yaw=-12, roll=3, z=12), None,     -4, 0.50, "minjerk", 0.45),  # look left
        (dict(pitch=-4, yaw=-10, roll=3, z=12),  [50, 50], -4, 0.22, "cartoon", 0.05),  # nod
        (dict(pitch=-12, yaw=0, roll=0, z=12),   [65, 65], 0,  0.45, "minjerk", 0.50),  # centre
    ],
    "hesitant": [
        (dict(pitch=10, yaw=6, roll=6, z=-8),    [-20, 15], 6,  1.0, "ease_in_out", 0.7),  # glance at listener
        (dict(pitch=18, yaw=24, roll=14, z=-10), [-40, 0],  14, 1.2, "ease_in_out", 1.1),  # look away, down
        (dict(pitch=14, yaw=18, roll=-6, z=-9),  None,      12, 0.9, "ease_in_out", 0.4),  # tilt of doubt
        (dict(pitch=15, yaw=20, roll=12, z=-10), [-30, 10], 12, 0.9, "ease_in_out", 1.3),  # still pause
        (dict(pitch=12, yaw=10, roll=8, z=-8),   [-15, 20], 8,  1.1, "ease_in_out", 0.5),  # half glance
        (dict(pitch=20, yaw=26, roll=16, z=-12), [-45, -5], 16, 1.2, "ease_in_out", 1.4),  # sink away
    ],
}


def _debate_animate_while_speaking(mini, condition: str, stop_event: threading.Event) -> None:
    from reachy_mini.utils import create_head_pose

    def move(head, antennas, body_deg, duration, method) -> None:
        kwargs = dict(
            head=create_head_pose(**head, degrees=True, mm=True),
            body_yaw=np.deg2rad(body_deg),
            duration=duration,
            method=method,
        )
        if antennas is not None:
            kwargs["antennas"] = np.deg2rad(antennas)
        mini.goto_target(**kwargs)

    try:
        cond = condition if condition in DEBATE_KEYFRAMES else "confident"
        move(*DEBATE_POSTURE[cond])
        while not stop_event.is_set():
            for head, antennas, body_deg, duration, method, pause in DEBATE_KEYFRAMES[cond]:
                if stop_event.is_set():
                    return
                move(head, antennas, body_deg, duration, method)
                if stop_event.wait(pause):
                    return
    except Exception:
        pass


def block_8_pygame_debate() -> None:
    """Contrôle manuel via Pygame du débat Autobahn (biaisé contre la limitation de vitesse)."""
    from reachy_mini import ReachyMini

    # Connect to the robot and prepare the audio BEFORE opening the window: a
    # window that is not being refreshed during these slow steps looks frozen
    # (spinning wheel on macOS).
    print("Préparation des répliques audio...")
    _preload_debate_audio()
    print("Connexion au robot (le daemon Reachy Mini doit être lancé)...")
    stack = contextlib.ExitStack()
    try:
        mini = stack.enter_context(ReachyMini(media_backend="no_media"))
    except ConnectionError as exc:
        print(f"Impossible de se connecter au robot : {exc}")
        print("Lance d'abord le daemon :  reachy-mini-daemon")
        return
    print("Robot connecté. Préparation de l'audio...")

    pygame.init()
    init_mixer_on_robot(frequency=22050, size=-16, channels=1, buffer=512)

    screen = pygame.display.set_mode((1040, 660))
    pygame.display.set_caption("Reachy Mini - Contrôle Manuel Débat Autobahn")
    font = pygame.font.SysFont(None, 22)
    font_bold = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

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
                pygame.mixer.music.set_volume(DEBATE_VOICE[cond]["volume"])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not run_stop.is_set():
                    time.sleep(0.05)
            else:
                status_msg = f"Audio introuvable pour : {last_label}"
        finally:
            was_stopped = run_stop.is_set()
            run_stop.set()
            if anim_thread is not None:
                # Longer than the longest single animation move so the
                # animation has finished before we command the neutral pose.
                anim_thread.join(timeout=DEBATE_MAX_MOVE_S + 0.3)
            _debate_go_neutral(mini)
            if not was_stopped and status_msg.startswith("Parole en cours"):
                status_msg = f"Terminé : {last_label}"
            busy = False

    with stack:
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
                                start_line(mini, item,
                                           item.get("force_condition", condition))
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
                "E5": (90, 120, 60),
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