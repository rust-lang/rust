//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:test-fail-static
//@ needs-subprocess

fn main() {
    panic!("test-fail-static");
}
