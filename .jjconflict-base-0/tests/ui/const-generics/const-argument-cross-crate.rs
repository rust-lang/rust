//@ run-pass
//@ revisions: full min
//@ aux-build:const_generic_lib.rs

extern crate const_generic_lib;

struct Container(#[allow(dead_code)] const_generic_lib::Alias);

fn main() {
    let res = const_generic_lib::function(const_generic_lib::Struct([14u8, 1u8, 2u8]));
    assert_eq!(res, 14u8);
    let _ = Container(const_generic_lib::Struct([0u8, 1u8]));
}
