fn main() {
    // foo must be used.
    foo();
}

// For this test to operate correctly, foo's body must start on exactly the same
// line and column and have the exact same length in bytes in a.rs and b.rs. In
// a.rs, the body must end on a line number which does not exist in b.rs.
// Basically, avoid modifying this file, including adding or removing whitespace!
fn foo() {
    assert_eq!(1, 1); ////
}
