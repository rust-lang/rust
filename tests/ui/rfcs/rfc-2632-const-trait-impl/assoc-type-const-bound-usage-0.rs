// FIXME(effects): Collapse the revisions into one once we support `<Ty as ~const Trait>::Proj`.
//@ revisions: unqualified qualified
//@[unqualified] check-pass
//@[qualified] known-bug: unknown

#![feature(const_trait_impl, effects)]

#[const_trait]
trait Trait {
    type Assoc: ~const Trait;
    fn func() -> i32;
}

#[cfg(unqualified)]
const fn unqualified<T: ~const Trait>() -> i32 {
    T::Assoc::func()
}

#[cfg(qualified)]
const fn qualified<T: ~const Trait>() -> i32 {
    <T as /* FIXME: ~const */ Trait>::Assoc::func()
}

fn main() {}
