// error-pattern:quux
fn my_err(s: str) -> ! { log(error, s); fail "quux"; }
fn main() { if my_err("bye") { } }
