// error-pattern: mismatched types

fn f(x: int) -> int { ret x; }

fn main() { spawn f(10); }