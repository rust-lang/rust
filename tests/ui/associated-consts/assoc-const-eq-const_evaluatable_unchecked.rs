// The impl of lint `const_evaluatable_unchecked` used to wrongly assume and `assert!` that
// successfully evaluating a type-system constant that has non-region args had to be an anon const.
// In the case below however we have a type-system assoc const (here: `<() as TraitA<T>>::K`).
//
// issue: <https://github.com/rust-lang/rust/issues/108220>
//@ check-pass
#![feature(associated_const_equality)]

pub trait TraitA<T> { const K: u8 = 0; }
pub trait TraitB<T> {}

impl<T> TraitA<T> for () {}
impl<T> TraitB<T> for () where (): TraitA<T, K = 0> {}

fn check<T>() where (): TraitB<T> {}

fn main() {}
