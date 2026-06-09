// https://github.com/rust-lang/rust/issues/140731
// This tests that there's no def path conflict between the
// remapped lifetime and the lifetime present in the source.

#![feature(impl_trait_in_assoc_type)]

trait Trait<'a> {}

impl<'a> Trait<'a> for u32 {
    type Opq2 = impl for<'a> Trait<'a>;
    //~^ ERROR: unconstrained opaque type
    //~| ERROR: type `Opq2` is not a member of trait `Trait`
    //~| ERROR: lifetime name `'a` shadows a lifetime name that is already in scope
}

fn main() {}
