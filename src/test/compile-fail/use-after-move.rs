// error-pattern:unsatisfied precondition constraint (for example, init(x
fn main() { let x = @5; let y <- x; log(debug, *x); }
