// error-pattern:fail

fn failfn() {
    fail;
}

fn main() {
    let x = @~0;
    failfn();
    log(error, x);
}