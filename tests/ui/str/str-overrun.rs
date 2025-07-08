//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let s: String = "hello".to_string();

    // Bounds-check panic.
    assert_eq!(s.as_bytes()[5], 0x0 as u8);
}
