// Exercise implicit trait object lifetime bounds inside
// inherent (generic) associated type paths.
// See also: <https://github.com/rust-lang/rust/issues/141997>

// FIXME: Ideally, this test would be check-pass. See below for details.

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

mod own { // the lifetime comes from the own generics
    struct Parent;
    impl Parent {
        type Ty<'a, T: 'a + super::AbideBy<'a> + ?Sized> = ();
    }

    // FIXME: Ideally, we would deduce `dyn Trait + 'r` from the bound `'a` on ty param `T` of
    //        inherent assoc ty `Ty` but for that we'd need to somehow obtain the resolution of the
    //        type-relative path `Parent::Ty` from HIR ty lowering in RBV.
    fn check<'r>() where Parent::Ty<'r, dyn super::Trait>: {}
    //~^ ERROR cannot deduce the lifetime bound for this trait object type from context
}

mod parent { // the lifetime comes from the parent generics
    struct Parent<'a>(&'a ());
    impl<'a> Parent<'a> {
        type Ty<T: 'a + super::AbideBy<'a> + ?Sized> = ();
    }

    // FIXME: Ideally, we would deduce `dyn Trait + 'r` from the bound `'a` on ty param `T` of
    //        inherent assoc ty `Ty` but for that we'd need to somehow obtain the resolution of the
    //        type-relative path `Parent<'r>::Ty` from HIR ty lowering in RBV.
    fn check<'r>() where Parent<'r>::Ty<dyn super::Trait>: {}
    //~^ ERROR cannot deduce the lifetime bound for this trait object type from context
}

trait Trait {}

// We use this to test that a given trait object lifetime bound is
// *exactly equal* to a given lifetime (not longer, not shorter).
trait AbideBy<'a> {}
impl<'a> AbideBy<'a> for dyn Trait + 'a {}

fn main() {}
