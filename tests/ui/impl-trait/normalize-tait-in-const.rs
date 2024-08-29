//@ known-bug: #103507

#![feature(type_alias_impl_trait)]
#![feature(const_trait_impl)]
#![feature(const_refs_to_cell)]

use std::marker::Destruct;

mod foo {
    trait T {
        type Item;
    }

    pub type Alias<'a> = impl T<Item = &'a ()>;

    struct S;
    impl<'a> T for &'a S {
        type Item = &'a ();
    }

    pub const fn filter_positive<'a>() -> &'a Alias<'a> {
        &&S
    }
}
use foo::*;

const fn with_positive<F: for<'a> ~const Fn(&'a Alias<'a>) + ~const Destruct>(fun: F) {
    fun(filter_positive());
}

const fn foo(_: &Alias<'_>) {}

const BAR: () = {
    with_positive(foo);
};

fn main() {}
