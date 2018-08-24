#![feature(extern_in_paths)]

use extern::ycrate; //~ ERROR can't find crate for `ycrate`

fn main() {}
