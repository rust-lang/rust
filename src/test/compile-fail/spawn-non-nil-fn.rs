// error-pattern: mismatched types

extern mod std;

fn main() { task::spawn(fn~() -> int { 10 }); }
