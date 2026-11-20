//! This is a regression test for <https://github.com/rust-lang/rust/issues/103507>.
//@ known-bug: #110395

#![feature(type_alias_impl_trait)]
#![feature(const_trait_impl, const_destruct)]

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

const fn with_positive<F: for<'a> [const] Fn(&'a Alias<'a>) + [const] Destruct>(fun: F) {
    fun(filter_positive());
}

const fn foo(_: &Alias<'_>) {}

const BAR: () = {
    with_positive(foo);
};

fn main() {}
