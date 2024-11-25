//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:test-fail-static
//@ ignore-emscripten no processes

fn main() {
    panic!("test-fail-static");
}
