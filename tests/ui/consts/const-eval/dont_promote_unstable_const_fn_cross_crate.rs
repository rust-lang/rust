//@ aux-build:stability.rs

extern crate stability;

use stability::foo;

fn main() {
    let _: &'static u32 = &foo(); //~ ERROR temporary value dropped while borrowed
    let _x: &'static u32 = &foo(); //~ ERROR temporary value dropped while borrowed
}
