//@ run-fail
//@ check-run-results: panicked
//@ check-run-results: test-assert-fmt 42 rust
//@ ignore-emscripten no processes

fn main() {
    assert!(false, "test-assert-fmt {} {}", 42, "rust");
}
