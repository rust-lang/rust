// check-pass
#![allow(dead_code)]
// Test that the requirement (in `Bar`) that `T::Bar : 'static` does
// not wind up propagating to `T`.

// pretty-expanded FIXME #23616

pub trait Foo {
    type Bar;

    fn foo(&self) -> Self;
}

pub struct Static<T:'static>(T);

struct Bar<T:Foo>
    where T::Bar : 'static
{
    x: Static<Option<T::Bar>>
}

fn main() { }
