// Test that error recovery in the parser to an EOF does not give an infinite
// spew of errors.

fn main() {
    let
} //~ ERROR expected pattern, found `}`
