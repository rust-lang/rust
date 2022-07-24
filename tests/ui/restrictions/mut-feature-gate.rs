// gate-test-mut_restriction
// compile-flags: --crate-type=lib
// revisions: with_gate without_gate
//[with_gate] check-pass

#![cfg_attr(with_gate, feature(mut_restriction))]

pub struct Foo {
    pub mut(self) alpha: u8, //[without_gate]~ ERROR
}
