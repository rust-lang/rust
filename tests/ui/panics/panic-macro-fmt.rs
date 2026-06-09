//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-fail-fmt 42 rust
//@ needs-subprocess

fn main() {
    panic!("test-fail-fmt {} {}", 42, "rust");
}
