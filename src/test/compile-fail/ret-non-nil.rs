// error-pattern: `return;` in function returning non-nil

fn f() { return; }

fn g() -> int { return; }

fn main() { f(); g(); }
