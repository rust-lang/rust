//! This test verifies the correct behavior of `std::mem::transmute` when converting
//! a value from a single-element array type (`[isize; 1]`) to a scalar integer type (`isize`).
//! This is a regression test ensuring that such transmutations, where source and destination
//! types have the same size and alignment but differ in their "immediacy" or structure,
//! are handled correctly.
//!
//! Regression test: <https://github.com/rust-lang/rust/issues/7988>

//@ run-pass

pub fn main() {
    unsafe {
        // Transmute a single-element array `[1]` (which might be treated as a "non-immediate" type)
        // to a scalar `isize` (an "immediate" type).
        // This is safe because `[isize; 1]` and `isize` have the same size and alignment.
        ::std::mem::transmute::<[isize; 1], isize>([1]);
    }
}
