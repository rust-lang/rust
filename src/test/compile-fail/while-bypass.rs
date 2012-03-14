// error-pattern: precondition constraint

fn f() -> int { let x: int; while 1 == 1 { x = 10; } ret x; }

fn main() { f(); }
