//@ check-pass

#![feature(const_trait_impl, generic_const_exprs)]
#![allow(incomplete_features)]

#[const_trait]
trait Trait {
    type Assoc: [const] Trait;
    fn func() -> i32;
}

struct Type<const N: i32>;

fn unqualified<T: const Trait>() -> Type<{ T::Assoc::func() }> {
    Type
}

fn qualified<T: const Trait>() -> Type<{ <T as Trait>::Assoc::func() }> {
    Type
}

fn main() {}
