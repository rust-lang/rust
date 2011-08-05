// error-pattern:assigning to immutable alias
// Make sure that nesting a block within a lambda doesn't let us
// mutate upvars from a lambda.
fn main() {
    let i = 0;
    let ctr = lambda() -> int {
        block() { i = i + 1; }();
        ret i;
    };
    log_err ctr();
    log_err ctr();
    log_err ctr();
    log_err ctr();
    log_err ctr();
    log_err i;
}
