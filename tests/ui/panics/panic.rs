//@ run-fail
//@ error-pattern:1 == 2
//@ ignore-emscripten no processes

fn main() {
    assert!(1 == 2);
}
