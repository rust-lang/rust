//@ run-fail
//@ error-pattern:assertion `left != right` failed: 1 + 1 definitely should not be 2
//@ error-pattern:  left: 2
//@ error-pattern: right: 2
//@ needs-subprocess

fn main() {
    assert_ne!(1 + 1, 2, "1 + 1 definitely should not be 2");
}
