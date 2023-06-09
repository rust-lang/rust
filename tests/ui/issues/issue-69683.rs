pub trait Element<S> {
    type Array;
}

impl<T> Element<()> for T {
    type Array = T;
}

impl<T: Element<S>, S> Element<[S; 3]> for T {
    type Array = [T::Array; 3];
}

trait Foo<I>
where
    u8: Element<I>,
{
    fn foo(self, x: <u8 as Element<I>>::Array);
}

impl<I> Foo<I> for u16
where
    u8: Element<I>,
{
    fn foo(self, _: <u8 as Element<I>>::Array) {}
}

fn main() {
    let b: [u8; 3] = [0u8; 3];

    0u16.foo(b); //~ ERROR type annotations needed
    //~^ ERROR type annotations needed
    //<u16 as Foo<[(); 3]>>::foo(0u16, b);
}
