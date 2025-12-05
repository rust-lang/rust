//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-fail-owned
//@ needs-subprocess

fn main() {
    panic!("test-fail-owned");
}
