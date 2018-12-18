#[test]
fn bar() {
    assert_eq!(4, 4);
}

// Having more than 1 test does seem to make a difference
// (i.e., this calls ptr::swap which having just one test does not).
#[test]
fn baz() {
    assert_eq!(5, 5);
}
