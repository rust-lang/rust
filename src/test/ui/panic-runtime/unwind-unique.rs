// run-fail
// error-pattern:explicit panic

fn failfn() {
    panic!();
}

fn main() {
    Box::new(0);
    failfn();
}
