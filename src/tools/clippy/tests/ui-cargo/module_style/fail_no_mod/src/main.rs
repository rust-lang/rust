#![warn(clippy::mod_module_files)]

mod bad;

fn main() {
    let _ = bad::Thing;
}
