//@ run-pass
//@ aux-build:macro_with_super_1.rs


#[macro_use]
extern crate macro_with_super_1;

declare!();

fn main() {
    bbb::ccc();
}
