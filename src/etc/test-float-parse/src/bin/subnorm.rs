use std::mem::transmute;
use test_float_parse::validate;

fn main() {
    for bits in 0u32..(1 << 21) {
        let single: f32 = unsafe { transmute(bits) };
        validate(&format!("{single:e}"));
        let double: f64 = unsafe { transmute(bits as u64) };
        validate(&format!("{double:e}"));
    }
}
