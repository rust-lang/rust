// error-pattern: mismatched types

fn add1(i: int) -> int { ret i + 1; }
fn main() { let f = @add1; let g = bind f(5); }
