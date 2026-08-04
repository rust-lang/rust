//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ build-pass
//@ compile-flags: --crate-type=lib
//@ edition: 2015

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
