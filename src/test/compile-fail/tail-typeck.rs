// error-pattern: mismatched types

fn f() -> int { be g(); }

fn g() -> uint { ret 0u; }

fn main() { let y = f(); }