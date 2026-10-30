//! Regression test for #153375.
//!
//! This checks that a region error involving an RPITIT from a trait
//! defined in another crate produces a proper diagnostic instead of
//! an ICE. When explaining the region error, we must be able
//! to resolve the lifetime parameters parent locally. The synthetic
//! params created for the impl side RPITIT associated type ensure
//! that their def parent is local, avoiding the ICE.

//@ aux-build: region-error-trait.rs
extern crate region_error_trait;
use region_error_trait::Request;

struct Bar<'ws>(&'ws ());

impl<'ws> Request for Bar<'ws> {
    type A<'a>
        = u8
    where
        Self: 'a;

    fn f(_: Self::A<'_>) -> impl Sized {}
    //~^ ERROR E0477
    //~| ERROR E0477
}

fn main() {}

//~? ERROR E0477
