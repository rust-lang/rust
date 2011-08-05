// error-pattern:assigning to immutable alias
// Make sure we can't write to upvars from lambdas
fn main() {
    let i = 0;
    let ctr = lambda() -> int {
        i = i + 1;
        ret i;
    };
    log_err ctr();
    log_err ctr();
    log_err ctr();
    log_err ctr();
    log_err ctr();
    log_err i;
}
