#![warn(clippy::mod_module_files)]

mod good;

fn main() {
    let _ = good::Thing;
}
