// FIXME(effects): Collapse the revisions into one once we support `<Ty as const Trait>::Proj`.
//@ revisions: unqualified qualified
//@[unqualified] check-pass
//@[qualified] known-bug: unknown

#![feature(const_trait_impl, effects, generic_const_exprs)]
#![allow(incomplete_features)]

#[const_trait]
trait Trait {
    type Assoc: ~const Trait;
    fn func() -> i32;
}

struct Type<const N: i32>;

#[cfg(unqualified)]
fn unqualified<T: const Trait>() -> Type<{ T::Assoc::func() }> {
    Type
}

#[cfg(qualified)]
fn qualified<T: const Trait>() -> Type<{ <T as /* FIXME: const */ Trait>::Assoc::func() }> {
    Type
}

fn main() {}
