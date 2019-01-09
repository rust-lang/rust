// aux-build:stability.rs

extern crate stability;

use stability::foo;

fn main() {
    let _: &'static u32 = &foo(); //~ ERROR does not live long enough
    let _x: &'static u32 = &foo(); //~ ERROR does not live long enough
}
