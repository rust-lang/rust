// check-pass

#![feature(min_type_alias_impl_trait)]
#![feature(type_alias_impl_trait)]
//~^ WARNING: the feature `type_alias_impl_trait` is incomplete

pub trait AssociatedImpl {
    type ImplTrait;

    fn f() -> Self::ImplTrait;
}

struct S<T>(T);

trait Associated {
    type A;
}

impl<'a, T: Associated<A = &'a ()>> AssociatedImpl for S<T> {
    type ImplTrait = impl core::fmt::Debug;

    fn f() -> Self::ImplTrait {
        ()
    }
}

fn main() {}
