// error-pattern:quux
fn my_err(s: str) -> ! { log_full(core::error, s); fail "quux"; }
fn main() { my_err("bye") == 3u; }
