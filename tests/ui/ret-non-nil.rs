// error-pattern: `return;` in a function whose return type is not `()`

fn f() { return; }

fn g() -> isize { return; }

fn main() { f(); g(); }
