// error-pattern: mismatched types

fn f() -> int { ret g(); }

fn g() -> uint { ret 0u; }

fn main() { let y = f(); }
