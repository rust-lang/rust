//@ compile-flags: -Z parse-crate-root-only
//@ check-pass

#![feature(const_trait_impl)]

struct S<
    T: for<'a> [const] Tr<'a> + 'static + [const] std::ops::Add,
    T: for<'a: 'b> [const] m::Trait<'a>,
>;
