//@ run-fail
//@ check-run-results:panicked
//@ check-run-results:test-assert-static
//@ ignore-emscripten no processes

fn main() {
    assert!(false, "test-assert-static");
}
