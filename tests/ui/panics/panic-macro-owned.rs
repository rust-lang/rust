//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-fail-owned
//@ ignore-emscripten no processes

fn main() {
    panic!("test-fail-owned");
}
