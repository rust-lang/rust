//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:test-fail-fmt 42 rust
//@ ignore-emscripten no processes

fn main() {
    panic!("test-fail-fmt {} {}", 42, "rust");
}
