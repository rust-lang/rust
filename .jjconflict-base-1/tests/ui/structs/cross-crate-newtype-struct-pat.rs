//@ run-pass
//@ aux-build:newtype_struct_xc.rs


extern crate newtype_struct_xc;

pub fn main() {
    let x = newtype_struct_xc::Au(21);
    match x {
        newtype_struct_xc::Au(n) => assert_eq!(n, 21)
    }
}
