//! Verify that we do not ICE when optimizing bodies with nonsensical bounds.
//@ compile-flags: -Copt-level=1
//@ edition: 2021
//@ build-pass

#![feature(trivial_bounds)]

async fn return_str() -> str
where
    str: Sized,
    //~^ WARN trait bound str: Sized does not depend on any type or lifetime parameters
{
    *"Sized".to_string().into_boxed_str()
}

fn main() {}
