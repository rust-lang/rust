//@ build-pass
//@ compile-flags: -Zmir-opt-level=3 --crate-type=lib

#![feature(trivial_bounds)]
#![allow(trivial_bounds)]

trait Foo {
    fn test(self);
}
fn baz<T>()
where
    &'static str: Foo,
{
    "Foo".test()
}
