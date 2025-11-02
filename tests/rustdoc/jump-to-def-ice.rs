// This test ensures that items with no body don't panic when generating
// jump to def links.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-ice.rs.html'

pub trait A {
    type T;
    type U;
}

impl A for () {
    type T = Self::U;
    type U = ();
}

pub trait C {
    type X;
}

pub struct F<T: C>(pub T::X);
