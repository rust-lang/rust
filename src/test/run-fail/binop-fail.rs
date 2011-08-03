// error-pattern:quux
fn my_err(s: str) -> ! { log_err s; fail "quux"; }
fn main() { my_err("bye") == 3u; }
