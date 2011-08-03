// error-pattern:quux
fn f() -> ! { fail "quux" }
fn g() -> int { alt f() { true { 1 } false { 0 } }; }
fn main() { g(); }
