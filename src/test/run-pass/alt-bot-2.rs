// n.b. This was only ever failing with optimization disabled.
fn a() -> int { alt ret 1 { 2 { 3 } } }
fn main() { a(); }
