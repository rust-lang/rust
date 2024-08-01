#![deny(elided_lifetimes_in_associated_constant)]
#![feature(generic_const_items)]
//~^ WARN the feature `generic_const_items` is incomplete

struct A;
impl A {
    const GAC_TYPE<T>: &str = "";
    const GAC_LIFETIME<'a>: &str = "";
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out
}

trait Trait {
    const GAC_TYPE<T>: &str = "";
    const GAC_LIFETIME<'a>: &str = "";
    //~^ ERROR missing lifetime specifier
}

fn main() {}
