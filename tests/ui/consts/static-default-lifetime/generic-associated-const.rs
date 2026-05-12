#![feature(generic_const_items)]

struct A;
impl A {
    const GAC_TYPE<T>: &str = "";
    const GAC_LIFETIME<'a>: &str = "";
    //~^ ERROR missing lifetime specifier
}

trait Trait {
    const GAC_TYPE<T>: &str = "";
    const GAC_LIFETIME<'a>: &str = "";
    //~^ ERROR missing lifetime specifier
}

fn main() {}
