// aux-build:msvc-data-only-lib.rs

extern crate msvc_data_only_lib;

fn main() {
    println!("The answer is {} !", msvc_data_only_lib::FOO);
}
