// error-pattern:assigning to upvar
// Make sure that nesting a block within a lambda doesn't let us
// mutate upvars from a lambda.
fn f2(x: block()) { x(); }

fn main() {
    let i = 0;
    let ctr = lambda () -> int { f2({|| i = i + 1; }); ret i; };
    log_full(core::error, ctr());
    log_full(core::error, ctr());
    log_full(core::error, ctr());
    log_full(core::error, ctr());
    log_full(core::error, ctr());
    log_full(core::error, i);
}
