//@ run-fail
//@ error-pattern: panicked
//@ error-pattern: test-assert-fmt 42 rust
//@ needs-subprocess

fn main() {
    assert!(false, "test-assert-fmt {} {}", 42, "rust");
}
