#![deny(elided_lifetimes_in_associated_constant)]

struct Foo<'a>(&'a ());

impl Foo<'_> {
    const STATIC: &str = "";
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out
}

trait Bar {
    const STATIC: &str;
}

impl Bar for Foo<'_> {
    const STATIC: &str = "";
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out
    //~| ERROR const not compatible with trait
}

fn main() {}
