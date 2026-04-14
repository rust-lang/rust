//! Beeper — generates a PCM tone and streams it to the sound driver.
//!
//! Opens `/dev/audio/card0/out0`, configures the stream with `AUDIO_SET_PARAMS`,
//! starts playback with `AUDIO_START`, and streams PCM frames via `vfs_write`.
//! Uses `poll(POLLOUT)` for backpressure.
#![no_std]
#![no_main]
use alloc::vec::Vec;
use core::default::Default;
extern crate alloc;

mod chime;
mod tone;

use abi::device::DeviceKind;
use abi::sound::{AudioParams, AudioSampleFormat, AudioStreamInfo, AUDIO_GET_INFO, AUDIO_SET_PARAMS, AUDIO_START};
use stem::{info, warn};

#[stem::main]
fn main(_arg: usize) -> ! {
    let tone_freq: Option<f64> = None;
    let seconds: f64 = 1.0;

    info!("Beeper: Waiting for /dev/audio/card0/out0 ...");

    // Wait for the audio driver to mount its VFS tree.
    let out_fd = loop {
        use abi::syscall::vfs_flags::O_RDWR;
        use stem::syscall::vfs::vfs_open;
        match vfs_open("/dev/audio/card0/out0", O_RDWR) {
            Ok(fd) => break fd,
            Err(_) => stem::time::sleep_ms(100),
        }
    };

    info!("Beeper: Opened /dev/audio/card0/out0 (fd={})", out_fd);

    // Query capabilities.
    let mut info_buf = AudioStreamInfo::default();
    {
        let call = abi::device::DeviceCall {
            kind: DeviceKind::Audio,
            op: AUDIO_GET_INFO,
            in_ptr: 0,
            in_len: 0,
            out_ptr: &mut info_buf as *mut AudioStreamInfo as u64,
            out_len: core::mem::size_of::<AudioStreamInfo>() as u32,
        };
        let _ = stem::syscall::vfs::vfs_device_call_raw(out_fd, &call);
    }

    // Request S16LE at 44100 Hz stereo.
    let desired = AudioParams {
        sample_format: AudioSampleFormat::S16LE as u32,
        rate: 44100,
        channels: 2,
        period_frames: 1024,
        buffer_frames: 4096,
        _reserved: [0; 3],
    };
    let mut accepted = AudioParams::default();
    {
        let call = abi::device::DeviceCall {
            kind: DeviceKind::Audio,
            op: AUDIO_SET_PARAMS,
            in_ptr: &desired as *const AudioParams as u64,
            in_len: core::mem::size_of::<AudioParams>() as u32,
            out_ptr: &mut accepted as *mut AudioParams as u64,
            out_len: core::mem::size_of::<AudioParams>() as u32,
        };
        let _ = stem::syscall::vfs::vfs_device_call_raw(out_fd, &call);
    }

    let sample_rate = accepted.rate;
    info!("Beeper: Configured stream (rate={}Hz, fmt={}, ch={})", sample_rate, accepted.sample_format, accepted.channels);

    // Start playback.
    {
        let call = abi::device::DeviceCall {
            kind: DeviceKind::Audio,
            op: AUDIO_START,
            in_ptr: 0,
            in_len: 0,
            out_ptr: 0,
            out_len: 0,
        };
        let _ = stem::syscall::vfs::vfs_device_call_raw(out_fd, &call);
    }

    // Generate PCM samples.
    let samples: Vec<u8> = if let Some(freq) = tone_freq {
        info!("Beeper: Generating {}Hz sine wave for {}s", freq, seconds);
        let mut gen = tone::ToneGenerator::new(freq, sample_rate as f64);
        let count = (seconds * sample_rate as f64) as usize;
        let mut v_i16 = Vec::with_capacity(count * 2);
        unsafe { v_i16.set_len(count * 2) };
        gen.fill_buffer(&mut v_i16);
        let mut v_u8 = Vec::with_capacity(count * 4);
        for s in v_i16 {
            v_u8.extend_from_slice(&s.to_le_bytes());
        }
        v_u8
    } else {
        info!("Beeper: Generating classic chime...");
        chime::generate_chime(sample_rate)
    };

    info!("Beeper: Playback started ({} bytes)", samples.len());

    let chunk_size = 4096usize;
    let mut offset = 0;

    while offset < samples.len() {
        let to_write = (samples.len() - offset).min(chunk_size);
        let buf = &samples[offset..offset + to_write];

        let mut sent = 0;
        while sent < buf.len() {
            use stem::syscall::vfs::vfs_write;
            match vfs_write(out_fd, &buf[sent..]) {
                Ok(0) | Err(_) => {
                    // No space — wait for POLLOUT.
                    let mut pollfds = [abi::syscall::PollFd {
                        fd: out_fd as i32,
                        events: abi::syscall::poll_flags::POLLOUT,
                        revents: 0,
                    }];
                    let _ = stem::syscall::vfs::vfs_poll(&mut pollfds, u64::MAX);
                }
                Ok(n) => { sent += n; }
            }
        }
        offset += to_write;
    }

    info!("Beeper: Finished.");
    loop {
        stem::time::sleep_ms(1000);
    }
}

