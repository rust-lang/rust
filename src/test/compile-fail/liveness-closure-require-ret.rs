fn force(f: fn() -> int) -> int { f() }
fn main() { log(debug, force(|| {})); } //~ ERROR mismatched types
