// error-pattern: In non-returning function f, some control paths may return
fn g() -> ! { fail; }
fn f() -> ! { ret 42; g(); }
fn main() { }
