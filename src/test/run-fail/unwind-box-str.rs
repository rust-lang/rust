// error-pattern:fail

fn failfn() {
    fail;
}

fn main() {
    let x = @~"hi";
    failfn();
    log(error, x);
}