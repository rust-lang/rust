// xfail-test After the closure syntax change this started failing with the wrong error message
// error-pattern: not all control paths return
fn force(f: fn() -> int) -> int { f() }
fn main() { log(error, force(|| {})); }
