// build-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

const fn foo<'a, 'b>() -> usize
where
    &'a (): Sized, &'b (): Sized,
{
    4
}

struct Foo<'a>(&'a ()) where for<'b> [u8; foo::<'a, 'b>()]: Sized;

fn main() {}
