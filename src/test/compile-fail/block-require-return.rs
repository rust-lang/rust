// error-pattern: not all control paths return
fn force(f: block() -> int) -> int { f() }
fn main() { log_err force({|| }); }
