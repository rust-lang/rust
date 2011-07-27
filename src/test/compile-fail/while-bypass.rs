// error-pattern: precondition constraint

fn f() -> int { let x: int; while true { x = 10; } ret x; }

fn main() { f(); }