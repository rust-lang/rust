//@ known-bug: rust-lang/rust#143896
#![feature(associated_const_equality)]
trait TraitA<'a> {
    const K: usize = 0;
}
impl<T> TraitA<'_> for () {}
impl dyn TraitA<'_> where (): TraitA<'a,K = 0> {}

pub fn main() {}
