#![feature(extern_in_paths)]

use extern::ycrate; //~ ERROR unresolved import `extern::ycrate`

fn main() {}
