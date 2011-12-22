// error-pattern:woe
fn f(a: int) { log_full(core::debug, a); }

fn main() { f(fail "woe"); }
