// error-pattern: precondition constraint

fn f() -> int { let x: int; ret x; }

fn main() { f(); }
