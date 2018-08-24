// error-pattern:fail

fn failfn() {
    panic!();
}

fn main() {
    Box::new(0);
    failfn();
}
