// check-pass
// aux-build:test-macros.rs

// Anonymize unstable non-dummy spans while still showing dummy spans `0..0`.
// normalize-stdout-test "bytes\([^0]\w*\.\.(\w+)\)" -> "bytes(LO..$1)"
// normalize-stdout-test "bytes\((\w+)\.\.[^0]\w*\)" -> "bytes($1..HI)"

#[macro_use]
extern crate test_macros;

print_bang! {

/**
*******
* DOC *
* DOC *
* DOC *
*******
*/
pub struct S;

}

fn main() {}
