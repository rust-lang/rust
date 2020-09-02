fn main() {
    // foo must be used.
    foo();
}

fn foo() {
    // foo's span in a.rs and b.rs must be identical
    // with respect to start line/column and length.
    assert_eq!(1, 1);////
}
