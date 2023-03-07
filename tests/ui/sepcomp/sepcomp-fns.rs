// run-pass
// compile-flags: -C codegen-units=3

// Test basic separate compilation functionality.  The functions should be able
// to call each other even though they will be placed in different compilation
// units.

// Generate some code in the first compilation unit before declaring any
// modules.  This ensures that the first module doesn't go into the same
// compilation unit as the top-level module.

fn one() -> usize { 1 }

mod a {
    pub fn two() -> usize {
        ::one() + ::one()
    }
}

mod b {
    pub fn three() -> usize {
        ::one() + ::a::two()
    }
}

fn main() {
    assert_eq!(one(), 1);
    assert_eq!(a::two(), 2);
    assert_eq!(b::three(), 3);
}
