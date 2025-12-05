//@ run-pass
//@ only-x86
//@ compile-flags: -Ctarget-feature=+sse2

use std::hint::black_box;

fn main() {
    let n: i64 = black_box(0x3fffffdfffffff);
    let r = f32::from_bits(0x5a7fffff);

    assert_ne!((n as f64) as f32, n as f32);

    // FIXME: these assertions fail if only x87 is enabled
    // see also https://github.com/rust-lang/rust/issues/114479
    assert_eq!(n as i64 as f32, r);
    assert_eq!(n as u64 as f32, r);
}
