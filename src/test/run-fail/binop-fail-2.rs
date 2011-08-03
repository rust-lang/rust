// error-pattern:quux
fn my_err(s: str) -> ! { log_err s; fail "quux"; }
fn main() { 3u == my_err("bye"); }
