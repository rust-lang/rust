//@ check-pass

#![feature(generic_const_items)]

struct A;
impl A {
    const GAC_TYPE<T>: &str = "";
    const GAC_LIFETIME<'a>: &str = "";
}

trait Trait {
    const GAC_TYPE<T>: &str = "";
    const GAC_LIFETIME<'a>: &str = "";
}

fn main() {}
