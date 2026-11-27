//@ run-fail
//@ error-pattern:assertion failed: 1 == 2
//@ needs-subprocess

fn main() {
    assert!(1 == 2);
}
