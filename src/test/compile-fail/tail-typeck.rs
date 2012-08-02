// error-pattern: mismatched types

fn f() -> int { return g(); }

fn g() -> uint { return 0u; }

fn main() { let y = f(); }
