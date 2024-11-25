//@ run-fail
//@ check-run-results:assertion failed: 1 == 2
//@ ignore-emscripten no processes

fn main() {
    assert!(1 == 2);
}
