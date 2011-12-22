// error-pattern:Unsatisfied precondition

fn main() {
    let j = fn () -> int { let i: int; ret i; }();
    log_full(core::error, j);
}
