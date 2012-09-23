// error-pattern: mismatched types

fn f(&&_x: int) {}
fn g(_a: fn(+v: int)) {}
fn main() { g(f); }
