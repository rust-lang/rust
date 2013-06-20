// Test that we can compile code that uses a `_` in function argument
// patterns.

fn foo((x, _): (int, int)) -> int {
    x
}

fn main() {
    assert_eq!(foo((22, 23)), 22);
}
