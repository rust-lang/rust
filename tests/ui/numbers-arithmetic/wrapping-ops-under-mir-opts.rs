//! Regression test for <https://github.com/rust-lang/rust/issues/82646>.

//@ run-pass
//@ compile-flags: -Zmir-opt-level=2 -Coverflow-checks=on

fn main() {
    assert_eq!(1_u32.wrapping_sub(2), u32::MAX);
    assert_eq!(u32::MAX.wrapping_add(2), 1);
    assert_eq!(i32::MIN.wrapping_sub(1), i32::MAX);
    assert_eq!(2_u32.wrapping_mul(u32::MAX), u32::MAX - 1);
}
