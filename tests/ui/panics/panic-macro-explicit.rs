//@ run-fail
//@ error-pattern:panicked
//@ error-pattern:explicit panic
//@ needs-subprocess

fn main() {
    panic!();
}
