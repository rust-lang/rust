//! Tone generation logic
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

pub struct ToneGenerator {
    frequency: f64,
    sample_rate: f64,
    phase: f64,
}

impl ToneGenerator {
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        Self {
            frequency,
            sample_rate,
            phase: 0.0,
        }
    }

    pub fn next_sample(&mut self) -> i16 {
        let value = libm::sin(self.phase * 2.0 * core::f64::consts::PI);
        let sample = (value * 32767.0) as i16;

        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        sample
    }

    pub fn fill_buffer(&mut self, buffer: &mut [i16]) {
        for i in (0..buffer.len()).step_by(2) {
            let sample = self.next_sample();
            buffer[i] = sample; // Left
            buffer[i + 1] = sample; // Right
        }
    }
}
