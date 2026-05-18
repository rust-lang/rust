// This test ensures that turbofish (`::<...>`) does not prevent jump-to-definition
// links from being generated.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/turbofish.rs.html'
use std::marker::PhantomData;

pub fn foo() {
    // `PhantomData::<usize>` — `PhantomData` must be linked despite the turbofish.
    type TheOne = PhantomData<()>;

    //@ has - '//a[@href="#13"]' 'TheOne'
    let _: TheOne::<usize> = PhantomData;
}
