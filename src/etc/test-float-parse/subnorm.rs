mod _common;

use std::mem::transmute;
use _common::validate;

fn main() {
    for bits in 0u32..(1 << 21) {
        let single: f32 = unsafe { transmute(bits) };
        validate(&format!("{:e}", single));
        let double: f64 = unsafe { transmute(bits as u64) };
        validate(&format!("{:e}", double));
    }
}
