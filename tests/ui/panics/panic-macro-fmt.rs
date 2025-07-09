//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    panic!("test-fail-fmt {} {}", 42, "rust");
}
