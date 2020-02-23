// error-pattern: arguments were supplied

fn f(x: isize) { }

fn main() { let i: (); i = f(); }
