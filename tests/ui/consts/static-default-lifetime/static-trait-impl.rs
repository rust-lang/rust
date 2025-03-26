#![deny(elided_lifetimes_in_associated_constant)]

trait Bar<'a> {
    const STATIC: &'a str;
}

struct A;
impl Bar<'_> for A {
    const STATIC: &str = "";
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out
    //~| ERROR lifetime parameters or bounds on const `STATIC` do not match the trait declaration
}

struct B;
impl Bar<'static> for B {
    const STATIC: &str = "";
}

fn main() {}
