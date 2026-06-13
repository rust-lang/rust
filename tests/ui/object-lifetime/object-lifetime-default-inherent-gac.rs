// Exercise implicit trait object lifetime bounds inside
// type-level inherent (generic) associated const paths.
// See also <https://github.com/rust-lang/rust/issues/141997>.

// FIXME: Ideally, this test would be check-pass. See below for details.

#![feature(min_generic_const_args, inherent_associated_types, generic_const_items)]
#![expect(incomplete_features)]

mod own { // the lifetime comes from the own generics
    struct Parent;
    impl Parent {
        type const CT<'a, T: 'a + super::AbideBy<'a> + ?Sized>: usize = 0;
    }

    // FIXME: Ideally, we would deduce `dyn Trait + 'r` from the bound `'a` on ty param `T` of
    //        type-level inherent assoc const `CT` but for that we'd need to somehow obtain the
    //        resolution of the type-relative path `Parent::CT` from HIR ty lowering in RBV.
    fn check<'r>() where [(); Parent::CT::<'r, dyn super::Trait>]: {}
    //~^ ERROR cannot deduce the lifetime bound for this trait object type from context
}

mod parent { // the lifetime comes from the parent generics
    struct Parent<'a>(&'a ());
    impl<'a> Parent<'a> {
        type const CT<T: 'a + super::AbideBy<'a> + ?Sized>: usize = 0;
    }

    //FIXME: Ideally, we would deduce `dyn Trait + 'r` from the bound `'a` on ty param `T` of
    //       type-level inherent assoc const `CT` but for that we'd need to somehow obtain the
    //       resolution of the type-relative path `Parent::<'r>::CT` from HIR ty lowering in RBV.
    fn check<'r>() where [(); Parent::<'r>::CT::<dyn super::Trait>]: {}
    //~^ ERROR cannot deduce the lifetime bound for this trait object type from context
}

trait Trait {}

// We use this to test that a given trait object lifetime bound is
// *exactly equal* to a given lifetime (not longer, not shorter).
trait AbideBy<'a> {}
impl<'a> AbideBy<'a> for dyn Trait + 'a {}

fn main() {}
