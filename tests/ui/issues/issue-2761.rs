//@ run-fail
//@ error-pattern:custom message
//@ needs-subprocess

fn main() {
    assert!(false, "custom message");
}
