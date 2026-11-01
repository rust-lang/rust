//! Regression test for <https://github.com/rust-lang/rust/issues/122638>.
//@ check-fail
#![feature(min_specialization)]
impl<'a, T: std::fmt::Debug, const N: usize> Iterator for ConstChunksExact<'a, T, { N }> {
    //~^ ERROR not all trait items implemented, missing: `Item` [E0046]
    fn next(&mut self) -> Option<Self::Item> {}
    //~^ ERROR mismatched types [E0308]
}
struct ConstChunksExact<'a, T: '_, const assert: usize> {}
//~^ ERROR `'_` cannot be used here [E0637]
//~| ERROR lifetime parameter `'a` is never used [E0392]
//~| ERROR type parameter `T` is never used [E0392]
impl<'a, T: std::fmt::Debug, const N: usize> Iterator for ConstChunksExact<'a, T, {}> {
    //~^ ERROR mismatched types [E0308]
    //~| ERROR the const parameter `N` is not constrained by the impl trait, self type, or predicates [E0207]
    type Item = &'a [T; N]; }
    //~^ ERROR: `Item` specializes an item from a parent `impl`, but that item is not marked `default`

fn main() {}
