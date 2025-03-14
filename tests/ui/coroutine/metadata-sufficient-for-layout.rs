// Check that the layout of a coroutine is available when auxiliary crate
// is compiled with --emit metadata.
//
// Regression test for #80998.
//
//@ aux-build:metadata-sufficient-for-layout.rs
//@ check-pass

#![feature(type_alias_impl_trait, rustc_attrs)]
#![feature(coroutine_trait)]

extern crate metadata_sufficient_for_layout;

mod helper {
    use std::ops::Coroutine;
    pub type F = impl Coroutine<(), Yield = (), Return = ()>;

    #[define_opaque(F)]
    fn f() -> F {
        metadata_sufficient_for_layout::g()
    }
}

// Static queries the layout of the coroutine.
static A: Option<helper::F> = None;

fn main() {}
