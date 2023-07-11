// gate-test-impl_restriction
// revisions: with_gate without_gate
//[with_gate] check-pass

#![crate_type = "lib"]
#![cfg_attr(with_gate, feature(impl_restriction))]

pub impl(crate) trait Bar {} //[without_gate]~ ERROR

mod foo {
    pub impl(in foo) trait Baz {} //[without_gate]~ ERROR
}
