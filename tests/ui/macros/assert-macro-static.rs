//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-assert-static
//@ ignore-emscripten no processes

fn main() {
    assert!(false, "test-assert-static");
}
