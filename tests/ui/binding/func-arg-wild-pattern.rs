//@ run-pass
// Test that we can compile code that uses a `_` in function argument
// patterns.


fn foo((x, _): (isize, isize)) -> isize {
    x
}

pub fn main() {
    assert_eq!(foo((22, 23)), 22);
}
