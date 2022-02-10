#![feature(negative_impls)]

// FIXME: this should compile

trait MyPredicate<'a> {}
impl<'a, T> !MyPredicate<'a> for &T where T: 'a {}
trait MyTrait<'a> {}
impl<'a, T: MyPredicate<'a>> MyTrait<'a> for T {}
impl<'a, T> MyTrait<'a> for &'a T {}
//~^ ERROR: conflicting implementations of trait `MyTrait<'_>` for type `&_`

fn main() {}
