#![deny(clippy::mod_module_files)]

mod good;
pub use dep_with_mod::with_mod::Thing;

fn main() {
    let _ = good::Thing;
    let _ = dep_with_mod::with_mod::Thing;
}
