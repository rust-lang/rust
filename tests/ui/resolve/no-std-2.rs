// run-pass

#![no_std]

extern crate std;

fn main() {
    let a = core::option::Option::Some("foo");
    a.unwrap();
}
