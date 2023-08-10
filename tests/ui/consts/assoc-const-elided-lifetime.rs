use std::marker::PhantomData;

struct Foo<'a> {
    x: PhantomData<&'a ()>,
}

impl<'a> Foo<'a> {
    const FOO: Foo<'_> = Foo { x: PhantomData::<&'a ()> };
    //~^ ERROR `'_` cannot be used here
}

fn main() {}
