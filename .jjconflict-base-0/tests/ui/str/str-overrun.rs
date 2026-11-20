//@ run-fail
//@ error-pattern:index out of bounds: the len is 5 but the index is 5
//@ needs-subprocess

fn main() {
    let s: String = "hello".to_string();

    // Bounds-check panic.
    assert_eq!(s.as_bytes()[5], 0x0 as u8);
}
