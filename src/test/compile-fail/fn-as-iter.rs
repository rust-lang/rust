// error-pattern:calling non-iter as sequence of for each loop
fn f() -> int { ret 4; }
fn main() { for each i in f() { } }
