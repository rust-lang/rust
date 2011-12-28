// error-pattern:expected `bool` but found `int`
// issue #516

fn main() { let x = true; let y = 1; let z = x + y; }
