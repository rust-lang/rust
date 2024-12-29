//@ run-fail
//@ check-run-results:assertion `left != right` failed: 1 + 1 definitely should not be 2
//@ check-run-results:  left: 2
//@ check-run-results: right: 2
//@ ignore-emscripten no processes

fn main() {
    assert_ne!(1 + 1, 2, "1 + 1 definitely should not be 2");
}
