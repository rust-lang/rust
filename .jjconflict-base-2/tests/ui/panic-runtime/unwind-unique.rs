//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

fn failfn() {
    panic!();
}

fn main() {
    Box::new(0);
    failfn();
}
