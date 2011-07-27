
// error-pattern: mismatched types

fn f(x: int) { }

fn main() { let i: (); i = f(()); }