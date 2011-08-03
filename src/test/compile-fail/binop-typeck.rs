// error-pattern:mismatched types
// issue #500

fn main() { let x = true; let y = 1; let z = x + y; }