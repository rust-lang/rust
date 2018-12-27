#![feature(extern_in_paths)]

use extern::xcrate::S; //~ ERROR unresolved import `extern::xcrate`

fn main() {}
