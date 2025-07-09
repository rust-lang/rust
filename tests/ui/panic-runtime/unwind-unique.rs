//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn failfn() {
    panic!();
}

fn main() {
    Box::new(0);
    failfn();
}
