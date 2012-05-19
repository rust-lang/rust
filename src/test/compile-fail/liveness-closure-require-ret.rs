// error-pattern: not all control paths return
fn force(f: fn() -> int) -> int { f() }
fn main() { log(error, force({|| })); }
