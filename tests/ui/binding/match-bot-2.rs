//@ run-pass
#![allow(unreachable_code)]
// n.b. This was only ever failing with optimization disabled.

fn a() -> isize { match return 1 { 2 => 3, _ => panic!() } }
pub fn main() { a(); }
