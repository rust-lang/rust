// revisions: good bad
//[good] check-pass

#![feature(return_position_impl_trait_in_trait)]

trait Foo<'a> {
    fn test(&'a self) -> impl Sized;
}

#[cfg(bad)]
impl<'a> Foo<'a> for () {
    fn test(&'a self) -> &'a Self { self }
    //[bad]~^ return type captures more lifetimes than trait definition
}

#[cfg(good)]
impl<'a> Foo<'a> for &'a () {
    fn test(&'a self) -> &'a Self { self }
}

fn main() {}
