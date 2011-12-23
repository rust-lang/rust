// error-pattern:assigning to upvar
// Make sure we can't write to upvars from lambdas
fn main() {
    let i = 0;
    let ctr = lambda () -> int { i = i + 1; ret i; };
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, i);
}
