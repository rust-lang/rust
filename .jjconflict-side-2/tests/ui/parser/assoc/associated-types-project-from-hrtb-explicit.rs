// Test you can't use a higher-ranked trait bound inside of a qualified
// path (just won't parse).

pub trait Foo<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

fn foo2<I>(x: <I as for<'x> Foo<&'x isize>>::A)
    //~^ ERROR expected identifier, found keyword `for`
    //~| ERROR expected one of `::` or `>`
{
}

pub fn main() {}
