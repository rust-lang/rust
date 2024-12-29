//@ run-fail
//@ check-run-results:assertion `left == right` failed: 1 + 1 definitely should be 3
//@ check-run-results:  left: 2
//@ check-run-results: right: 3
//@ ignore-emscripten no processes

fn main() {
    assert_eq!(1 + 1, 3, "1 + 1 definitely should be 3");
}
