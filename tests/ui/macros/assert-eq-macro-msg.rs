//@ run-fail
//@ error-pattern:assertion `left == right` failed: 1 + 1 definitely should be 3
//@ error-pattern:  left: 2
//@ error-pattern: right: 3
//@ needs-subprocess

fn main() {
    assert_eq!(1 + 1, 3, "1 + 1 definitely should be 3");
}
