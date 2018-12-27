// aux-build:ufuncs.rs

extern crate ufuncs;

use ufuncs::udrop;

fn main() {
    udrop as fn([u8]);
    //~^ERROR E0277
}
