// error-pattern:Unsatisfied precondition
// xfail-stage0

fn main() { let j = fn () -> int { let i: int; ret i; }(); log_err j; }