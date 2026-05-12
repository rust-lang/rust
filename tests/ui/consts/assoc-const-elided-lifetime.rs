use std::marker::PhantomData;

struct Foo<'a> {
    x: PhantomData<&'a ()>,
}

impl<'a> Foo<'a> {
    const FOO: Foo<'_> = Foo { x: PhantomData::<&()> };
    //~^ ERROR missing lifetime specifier

    const BAR: &() = &();
    //~^ ERROR missing lifetime specifier
}

fn main() {}
