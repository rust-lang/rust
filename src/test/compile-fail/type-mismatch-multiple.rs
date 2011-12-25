// Checking that the compiler reports multiple type errors at once
// error-pattern:mismatched types: expected 'bool'
// error-pattern:mismatched types: expected 'int'

fn main() { let a: bool = 1; let b: int = true; }
