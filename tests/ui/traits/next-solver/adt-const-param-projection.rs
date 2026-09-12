//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ build-pass
//@ compile-flags: --crate-type=lib
//@ edition: 2015

// Regression test for https://github.com/rust-lang/rust/issues/156294.
// We used to not normalize the type we get back from const evaluation, so the value of
// `EMPTY_MATRIX` had the type `<Type as Trait>::Matrix` instead of `[usize; 1]`. Nobody
// normalized it later on either, so we ended up ICEing when mangling the symbol name of
// `Walk::<EMPTY_MATRIX>::new`.

#![feature(adt_const_params)]

pub const EMPTY_MATRIX: <Type as Trait>::Matrix = [1];
pub struct Walk<const REMAINING: <Type as Trait>::Matrix>;
impl Walk<EMPTY_MATRIX> {
    pub fn new() {}
}
pub enum Type {}
pub trait Trait {
    type Matrix;
}
impl Trait for Type {
    type Matrix = [usize; 1];
}
