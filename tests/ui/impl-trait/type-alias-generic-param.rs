// Regression test for issue #59342
// Checks that we properly detect defining uses of opaque
// types in 'item' position when generic parameters are involved
//
//@ run-pass
#![feature(impl_trait_in_assoc_type)]

trait Meow { //~ WARN trait `Meow` is never used
    type MeowType;
    fn meow(self) -> Self::MeowType;
}

impl<T, I> Meow for I
where
    I: Iterator<Item = T>,
{
    type MeowType = impl Iterator<Item = T>;
    fn meow(self) -> Self::MeowType {
        self
    }
}

fn main() {}
