//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:test-fail-owned
//@ ignore-emscripten no processes

fn main() {
    panic!("test-fail-owned");
}
