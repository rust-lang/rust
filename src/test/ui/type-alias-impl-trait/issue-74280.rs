// Regression test for #74280.

#![feature(type_alias_impl_trait)]

type Test = impl Copy;

fn test() -> Test {
    let y = || -> Test { () };
    //~^ ERROR: concrete type differs from previous defining opaque type use
    7
}

fn main() {}
