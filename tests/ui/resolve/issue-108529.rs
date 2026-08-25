#![allow(nonstandard_style)]
use f::f::f; //~ ERROR

trait f {
    extern "C" fn f();
}

fn main() {}
