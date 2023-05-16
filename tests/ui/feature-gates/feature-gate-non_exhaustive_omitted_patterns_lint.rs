// check-fail

#![deny(non_exhaustive_omitted_patterns)]
//~^ WARNING unknown lint: `non_exhaustive_omitted_patterns`
//~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
//~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
#![allow(non_exhaustive_omitted_patterns)]
//~^ WARNING unknown lint: `non_exhaustive_omitted_patterns`
//~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
//~| WARNING unknown lint: `non_exhaustive_omitted_patterns`

fn main() {
    enum Foo {
        A,
        B,
        C,
    }

    #[allow(non_exhaustive_omitted_patterns)]
    //~^ WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    match Foo::A {
        _ => {}
    }

    #[warn(non_exhaustive_omitted_patterns)]
    //~^ WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    //~| WARNING unknown lint: `non_exhaustive_omitted_patterns`
    match Foo::A {
        _ => {}
    }

    match Foo::A {}
    //~^ ERROR non-exhaustive patterns
}
