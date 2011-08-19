// error-pattern:woe
fn f(a: int) { log a; }

fn main() { f(fail "woe"); }
