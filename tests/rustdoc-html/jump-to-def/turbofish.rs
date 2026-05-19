// This test ensures that turbofish (`::<...>`) does not prevent jump-to-definition
// links from being generated.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/turbofish.rs.html'
use std::marker::PhantomData as TheOne;


pub fn foo() {
    //@ has - '//a[@href="{{channel}}/core/marker/struct.PhantomData.html"]' 'TheOne'
    let _: TheOne::<usize>;
}
