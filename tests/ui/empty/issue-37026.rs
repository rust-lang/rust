//@ aux-build:empty-struct.rs

extern crate empty_struct;

fn main() {
    let empty_struct::XEmpty2 = (); //~ ERROR mismatched types
    let empty_struct::XEmpty6(..) = (); //~ ERROR mismatched types
}
