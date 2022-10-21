// gate-test-impl_restriction
// compile-flags: --crate-type=lib
// revisions: with_gate without_gate
//[with_gate] check-pass

#![cfg_attr(with_gate, feature(impl_restriction))]

pub impl(crate) trait Bar {} //[without_gate]~ ERROR

mod foo {
    pub impl(in foo) trait Baz {} //[without_gate]~ ERROR
}
