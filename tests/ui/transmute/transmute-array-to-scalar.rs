//! Verify transmuting from a single-element array to a scalar is allowed.
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
