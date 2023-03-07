// revisions: stock with_negative_coherence
//[with_negative_coherence] check-pass

#![feature(negative_impls)]
#![cfg_attr(with_negative_coherence, feature(with_negative_coherence))]

trait MyPredicate<'a> {}

impl<'a, T> !MyPredicate<'a> for &'a T where T: 'a {}

trait MyTrait<'a> {}

impl<'a, T: MyPredicate<'a>> MyTrait<'a> for T {}
impl<'a, T> MyTrait<'a> for &'a T {}
//[stock]~^ ERROR: conflicting implementations of trait `MyTrait<'_>` for type `&_`

fn main() {}
