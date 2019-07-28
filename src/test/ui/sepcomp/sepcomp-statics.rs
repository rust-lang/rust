// run-pass
#![allow(dead_code)]
// compile-flags: -C codegen-units=3

// Test references to static items across compilation units.


fn pad() -> usize { 0 }

const ONE: usize = 1;

mod b {
    // Separate compilation always switches to the LLVM module with the fewest
    // instructions.  Make sure we have some instructions in this module so
    // that `a` and `b` don't go into the same compilation unit.
    fn pad() -> usize { 0 }

    pub static THREE: usize = ::ONE + ::a::TWO;
}

mod a {
    fn pad() -> usize { 0 }

    pub const TWO: usize = ::ONE + ::ONE;
}

fn main() {
    assert_eq!(ONE, 1);
    assert_eq!(a::TWO, 2);
    assert_eq!(b::THREE, 3);
}
