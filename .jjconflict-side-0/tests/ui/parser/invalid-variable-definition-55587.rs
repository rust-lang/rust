// https://github.com/rust-lang/rust/issues/55587
use std::path::Path;

fn main() {
    let Path::new(); //~ ERROR expected tuple struct or tuple variant
}
