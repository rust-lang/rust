//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    assert_ne!(1 + 1, 2, "1 + 1 definitely should not be 2");
}
