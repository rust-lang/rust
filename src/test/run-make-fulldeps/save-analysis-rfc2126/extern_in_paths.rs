#![feature(extern_in_paths)]

use extern::krate2;

fn main() {
    extern::krate2::hello();
}
