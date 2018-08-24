// error-pattern: mismatched types

fn f(x: isize) { }

fn main() { let i: (); i = f(()); }
