//@check-pass

#![warn(clippy::path_ends_with_ext)]

use std::path::Path;

fn f(p: &Path) {
    p.ends_with(".dot");
}

fn main() {}
