#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use alloc::vec::Vec;
use core::f32::consts::PI;

pub fn generate_chime(sample_rate: u32) -> Vec<u8> {
    let duration_secs = 10.0; // Sped up ambient swell (300%)
    let total_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut buffer = Vec::with_capacity(total_samples * 4);

    // Base Frequencies (A3 Major - warmer, less whistle-y)
    let f_root = 220.0; // A3
    let f_third = 277.18; // C#4
    let f_fifth = 329.63; // E4
    let f_octave = 440.0; // A4 (shimmer)

    // Detuning for "Air/Chorus" effect
    // We mix multiple sines per note to break the perfect interference patterns
    let oscs = [
        (f_root, 0.4),
        (f_root * 1.005, 0.3), // Root + slight detune
        (f_third, 0.3),
        (f_third * 0.997, 0.2), // Third + slight detune
        (f_fifth, 0.3),
        (f_fifth * 1.004, 0.2), // Fifth
        (f_octave, 0.1),        // Quiet octave
    ];

    let attack = 0.5; // Quick fade-in so chime is heard immediately
    let release = 2.66; // Sped up tail

    // Pre-calc envelope points
    let release_start_sample = (total_samples as f32 * 0.6) as usize; // Check later
    let release_len_samples = (sample_rate as f32 * release) as usize;
    let attack_samples = (sample_rate as f32 * attack) as usize;

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;

        // Ethereal Envelope
        // 0 -> Attack -> Hold -> Slow Release
        let mut env = 1.0;
        if i < attack_samples {
            env = t / attack;
            // Quadratic ease-in for softer start (remains quiet longer)
            env = env * env;
        } else if t > (duration_secs - release) {
            let r_t = (t - (duration_secs - release)) / release;
            if r_t >= 1.0 {
                env = 0.0;
            } else {
                env = 1.0 - r_t;
                // Exponential decay for tail
                env = env * env;
            }
        }

        // Sped up "Breathing" Tremolo (1.5 Hz)
        let breath = 0.9 + 0.1 * libm::sinf(2.0 * PI * 1.5 * t);

        // Sum oscillators
        let mut signal = 0.0;
        for (freq, amp) in oscs.iter() {
            signal += libm::sinf(2.0 * PI * freq * t) * amp;
        }

        signal *= env * breath * 0.15; // Master gain

        // Soft Clipping / Saturation to warm it up
        let signal = if signal > 0.8 {
            0.8 + (signal - 0.8) * 0.5
        } else {
            signal
        };

        let sample_l = signal;
        // Stereo widener: Phase shift the right channel slightly
        let sample_r = signal * 0.9 + 0.1 * libm::sinf(2.0 * PI * (f_root * 1.01) * t) * env * 0.15;

        let pcm_l = (sample_l * 30000.0) as i16;
        let pcm_r = (sample_r * 30000.0) as i16;

        buffer.extend_from_slice(&pcm_l.to_le_bytes());
        buffer.extend_from_slice(&pcm_r.to_le_bytes());
    }

    buffer
}
