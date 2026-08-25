//! Ensure shift operations don't mutate their right operand.
//!
//! This test checks that expressions like `0 << b` don't accidentally
//! modify the variable `b` due to codegen issues with virtual registers.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/152>.

//@ run-pass

pub fn main() {
    let mut b: usize = 1;
    while b < size_of::<usize>() {
        // This shift operation should not mutate `b`
        let _ = 0_usize << b;
        b <<= 1;
        std::hint::black_box(b);
    }
    assert_eq!(size_of::<usize>(), b);
}
