use std::io;
use std::io::prelude::*;
use std::mem::transmute;

// Nothing up my sleeve: Just (PI - 3) in base 16.
#[allow(dead_code)]
pub const SEED: [u32; 3] = [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e];

pub fn validate(text: &str) {
    let mut out = io::stdout();
    let x: f64 = text.parse().unwrap();
    let f64_bytes: u64 = unsafe { transmute(x) };
    let x: f32 = text.parse().unwrap();
    let f32_bytes: u32 = unsafe { transmute(x) };
    writeln!(&mut out, "{:016x} {:08x} {}", f64_bytes, f32_bytes, text).unwrap();
}
