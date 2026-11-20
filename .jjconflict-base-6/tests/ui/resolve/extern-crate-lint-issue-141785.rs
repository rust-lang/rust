//@ check-pass
//@ edition:2018

#![no_implicit_prelude]
#![warn(unused_extern_crates)]

extern crate std;
fn main() {
    let r = 1u16..10;
    std::println!("{:?}", r.is_empty());
}
