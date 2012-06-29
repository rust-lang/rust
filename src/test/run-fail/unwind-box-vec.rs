// error-pattern:fail

fn failfn() {
    fail;
}

fn main() {
    let x = @~[0, 1, 2, 3, 4, 5];
    failfn();
    log(error, x);
}