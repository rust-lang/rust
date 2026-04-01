//@ compile-flags: -Z parse-crate-root-only

#![feature(const_trait_impl)]

struct S<T: [const] [const] Tr>;
//~^ ERROR expected identifier, found `]`
