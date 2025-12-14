//@ edition:2021
//@ aux-crate:to_reuse_functions=to-reuse-functions.rs
//@ pretty-mode:hir
//@ pretty-compare-only
//@ pp-exact:delegation-inherit-attributes.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]

extern crate to_reuse_functions;

mod to_reuse {
    #[must_use = "foo: some reason"]
    #[cold]
    pub fn foo(x: usize) -> usize {
        x
    }

    #[must_use]
    #[cold]
    pub fn foo_no_reason(x: usize) -> usize {
        x
    }

    #[cold]
    #[deprecated]
    pub fn bar(x: usize) -> usize {
        x
    }
}

#[deprecated]
reuse to_reuse::foo as foo1 {
    self + 1
}

reuse to_reuse::foo_no_reason {
    self + 1
}

#[deprecated]
#[must_use = "some reason"]
reuse to_reuse::foo as foo2 {
    self + 1
}

reuse to_reuse::bar;

reuse to_reuse_functions::unsafe_fn_extern;
reuse to_reuse_functions::extern_fn_extern;
reuse to_reuse_functions::const_fn_extern;
#[must_use = "some reason"]
reuse to_reuse_functions::async_fn_extern;


fn main() {}
