// error-pattern:fail

fn failfn() {
    fail;
}

trait i {
    fn foo();
}

impl of i for ~int {
    fn foo() { }
}

fn main() {
    let x = ~0 as i;
    failfn();
    log(error, x);
}