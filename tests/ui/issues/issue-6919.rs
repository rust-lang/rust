// run-pass
#![allow(unused_attributes)]
// aux-build:iss.rs

// pretty-expanded FIXME #23616

extern crate issue6919_3;

pub fn main() {
    let _ = issue6919_3::D.k;
}
