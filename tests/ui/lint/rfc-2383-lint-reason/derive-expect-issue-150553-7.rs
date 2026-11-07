// The `#[expect]` sharing with derive-generated code also applies when the
// derive itself is expanded from a macros 2.0 macro: the impl is still created
// by the innermost (derive) expansion, so it is recognized as derive-generated
// no matter what macro produced the derived item. A genuinely unfulfilled
// expectation is still reported exactly once.

//@ check-pass

#![feature(decl_macro)]
#![deny(redundant_lifetimes)]

use std::fmt::Debug;

macro fulfilled() {
    #[derive(Debug)]
    #[expect(redundant_lifetimes)]
    pub struct RefWrapper<'a, T>
    where
        'a: 'static,
        T: Debug,
    {
        pub t_ref: &'a T,
    }
}

fulfilled!();

macro passthrough($i:item) {
    $i
}

passthrough! {
    #[derive(Debug)]
    #[expect(redundant_lifetimes)]
    pub struct RefWrapperPassthrough<'a, T>
    where
        'a: 'static,
        T: Debug,
    {
        pub t_ref: &'a T,
    }
}

macro unfulfilled() {
    #[derive(Debug)]
    #[expect(unexpected_cfgs)]
    //~^ WARN this lint expectation is unfulfilled
    pub struct Unfulfilled {
        pub x: i64,
        #[cfg(false)]
        pub gone: u8,
    }
}

unfulfilled!();

fn main() {}
