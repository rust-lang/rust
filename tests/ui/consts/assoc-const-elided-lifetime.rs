#![deny(elided_lifetimes_in_associated_constant)]

use std::marker::PhantomData;

struct Foo<'a> {
    x: PhantomData<&'a ()>,
}

impl<'a> Foo<'a> {
    const FOO: Foo<'_> = Foo { x: PhantomData::<&()> };
    //~^ ERROR `'_` cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    const BAR: &() = &();
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
