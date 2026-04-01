#![feature(marker_trait_attr)]

#[marker]
trait A {}
impl<'a> A for (&'static (), &'a ()) {} //~ ERROR type annotations needed
impl<'a> A for (&'a (), &'static ()) {} //~ ERROR type annotations needed

fn main() {}
