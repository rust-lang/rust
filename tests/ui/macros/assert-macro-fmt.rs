//@ run-fail
//@ error-pattern: panicked
//@ error-pattern: test-assert-fmt 42 rust
//@ ignore-emscripten no processes

fn main() {
    assert!(false, "test-assert-fmt {} {}", 42, "rust");
}
