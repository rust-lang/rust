//@ run-fail
//@ check-run-results:assertion `left != right` failed
//@ check-run-results:  left: 14
//@ check-run-results: right: 14
//@ ignore-emscripten no processes

fn main() {
    assert_ne!(14, 14);
}
