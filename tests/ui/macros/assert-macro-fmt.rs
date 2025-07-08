//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    assert!(false, "test-assert-fmt {} {}", 42, "rust");
}
