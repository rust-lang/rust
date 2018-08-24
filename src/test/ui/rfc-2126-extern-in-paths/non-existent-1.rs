#![feature(extern_in_paths)]

use extern::xcrate::S; //~ ERROR can't find crate for `xcrate`

fn main() {}
