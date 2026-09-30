//@ check-pass
//@ compile-flags: --crate-type=lib

#![feature(gca_min_const_items)]
#![feature(fn_delegation)]
#![feature(adt_const_params)]
#![feature(unsized_const_params)]

trait Trait<'a, T, const N: str> {
    fn foo<'v, A, B>(&self) {}
}
reuse Trait::foo;
