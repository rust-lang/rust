// error-pattern:assigning to immutable alias

fn f(i: &int) { i += 2; }

fn main() { f(1); }
