//@ check-pass
//@ compile-flags: -Zmir-opt-level=3

#![feature(type_alias_impl_trait)]

mod foo {
    pub trait T {
        type Item;
    }

    pub type Alias<'a> = impl T<Item = &'a ()>;

    struct S;
    impl<'a> T for &'a S {
        type Item = &'a ();
    }

    pub fn filter_positive<'a>() -> Alias<'a> {
        &S
    }
}

use foo::*;

fn with_positive(fun: impl Fn(Alias<'_>)) {
    fun(filter_positive());
}

fn main() {
    with_positive(|_| ());
}
