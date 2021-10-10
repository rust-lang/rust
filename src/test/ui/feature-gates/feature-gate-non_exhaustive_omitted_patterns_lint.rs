#![deny(non_exhaustive_omitted_patterns)]
//~^ ERROR the `non_exhaustive_omitted_patterns` lint is unstable
//~| ERROR the `non_exhaustive_omitted_patterns` lint is unstable
#![allow(non_exhaustive_omitted_patterns)]
//~^ ERROR the `non_exhaustive_omitted_patterns` lint is unstable
//~| ERROR the `non_exhaustive_omitted_patterns` lint is unstable

fn main() {
    enum Foo {
        A, B, C,
    }

    #[allow(non_exhaustive_omitted_patterns)]
    match Foo::A {
        Foo::A => {}
        Foo::B => {}
    }
    //~^^^^^ ERROR the `non_exhaustive_omitted_patterns` lint is unstable
    //~| ERROR the `non_exhaustive_omitted_patterns` lint is unstable
    //~| ERROR the `non_exhaustive_omitted_patterns` lint is unstable
    //~| ERROR the `non_exhaustive_omitted_patterns` lint is unstable

    match Foo::A {
        Foo::A => {}
        Foo::B => {}
        #[warn(non_exhaustive_omitted_patterns)]
        _ => {}
    }
    //~^^^ ERROR the `non_exhaustive_omitted_patterns` lint is unstable
    //~| ERROR the `non_exhaustive_omitted_patterns` lint is unstable
}
