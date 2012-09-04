// n.b. This was only ever failing with optimization disabled.
fn a() -> int { match return 1 { 2 => 3, _ => fail } }
fn main() { a(); }
