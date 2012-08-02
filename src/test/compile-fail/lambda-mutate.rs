// error-pattern:assigning to captured outer variable in a heap closure
// Make sure we can't write to upvars from fn@s
fn main() {
    let i = 0;
    let ctr = fn@ () -> int { i = i + 1; return i; };
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, i);
}
