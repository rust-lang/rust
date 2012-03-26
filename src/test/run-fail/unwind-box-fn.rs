// error-pattern:fail

fn failfn() {
    fail;
}

fn main() {
    let y = ~0;
    let x = @fn@() {
        log(error, y);
    };
    failfn();
    log(error, x);
}