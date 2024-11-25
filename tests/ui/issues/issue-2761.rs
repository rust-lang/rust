//@ run-fail
//@ check-run-results:custom message
//@ ignore-emscripten no processes

fn main() {
    assert!(false, "custom message");
}
