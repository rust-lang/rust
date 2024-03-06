//@ run-fail
//@ error-pattern:assertion `left == right` failed
//@ error-pattern:  left: 14
//@ error-pattern: right: 15
//@ ignore-emscripten no processes

fn main() {
    assert_eq!(14, 15);
}
