// compile-flags: -Z unstable-options

#![deny(rustc::vec_new)]

fn main() {
    let _: Vec<u8> = Vec::new(); //~ ERROR usage of `Vec::new()`
    let _: Vec<u8> = vec![];
}
