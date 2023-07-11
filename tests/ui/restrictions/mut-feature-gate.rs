// gate-test-mut_restriction
// revisions: with_gate without_gate
//[with_gate] check-pass

#![crate_type = "lib"]
#![cfg_attr(with_gate, feature(mut_restriction))]

pub struct Foo {
    pub mut(self) alpha: u8, //[without_gate]~ ERROR
}
