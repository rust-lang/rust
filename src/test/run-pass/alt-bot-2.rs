// n.b. This was only ever failing with optimization disabled.
fn a() -> int { alt check return 1 { 2 => 3 } }
fn main() { a(); }
