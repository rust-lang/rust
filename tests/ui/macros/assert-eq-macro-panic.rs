//@ run-fail
//@ error-pattern:assertion `left == right` failed
//@ error-pattern:  left: 14
//@ error-pattern: right: 15
//@ needs-subprocess

fn main() {
    assert_eq!(14, 15);
}
