pub trait Foo: Sized {
    const SIZE: usize;

    fn new(slice: &[u8; Foo::SIZE]) -> Self;
    //~^ ERROR: type annotations needed: cannot resolve `_: Foo`
}

pub struct Bar<T: ?Sized>(T);

impl Bar<[u8]> {
    const SIZE: usize = 32;

    fn new(slice: &[u8; Self::SIZE]) -> Self {
        Foo(Box::new(*slice)) //~ ERROR: expected function, found trait `Foo`
    }
}

fn main() {}
