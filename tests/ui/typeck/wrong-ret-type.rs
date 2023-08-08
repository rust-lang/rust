// error-pattern: mismatched types
fn mk_int() -> usize { let i: isize = 3; return i; }
fn main() { }
