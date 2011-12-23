// error-pattern:woe
fn f(a: int) { log(debug, a); }

fn main() { f(fail "woe"); }
