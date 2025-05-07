//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-assert-static
//@ needs-subprocess

fn main() {
    assert!(false, "test-assert-static");
}
