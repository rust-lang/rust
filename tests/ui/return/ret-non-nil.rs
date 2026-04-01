fn f() { return; }

fn g() -> isize { return; } //~ ERROR `return;` in a function whose return type is not `()`

fn main() { f(); g(); }
