// -*- rust -*-
// error-pattern: mismatched types

fn f(x: &int) { log_err x; }
fn h(x: int) { log_err x; }
fn main() { let g: fn(int)  = f; g(10); g = h; g(10); }


