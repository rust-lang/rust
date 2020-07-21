#![feature(non_ascii_idents)]

extern crate ьаг; //~ ERROR cannot load a crate with a non-ascii name `ьаг`
//~| ERROR can't find crate for `ьаг`

fn main() {}
