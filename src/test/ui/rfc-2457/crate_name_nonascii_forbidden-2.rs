// compile-flags:--extern му_сгате
// edition:2018
#![feature(non_ascii_idents)]

use му_сгате::baz; //~  ERROR cannot load a crate with a non-ascii name `му_сгате`

fn main() {}
