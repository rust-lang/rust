// Regression test for issue #59342
// Checks that we properly detect defining uses of opaque
// types in 'item' position when generic parameters are involved
//
// run-pass
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Meow {
    type MeowType;
    fn meow(self) -> Self::MeowType;
}

impl<T, I> Meow for I
    where I: Iterator<Item = T>
{
    type MeowType = impl Iterator<Item = T>;
    fn meow(self) -> Self::MeowType {
        self
    }
}

fn main() {}
