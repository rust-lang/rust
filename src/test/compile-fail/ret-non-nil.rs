// error-pattern: ret; in function returning non-nil

fn f() { ret; }

fn g() -> int { ret; }

fn main() { f(); g(); }