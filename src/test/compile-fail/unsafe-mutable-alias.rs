// xfail-stage0
// error-pattern:mutable alias to a variable that roots another alias

fn f(a: &int, b: &mutable int) -> int { b += 1; ret a + b; }

fn main() { let i = 4; log f(i, i); }