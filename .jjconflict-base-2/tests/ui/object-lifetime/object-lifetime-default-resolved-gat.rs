// Check that resolved GAT paths correctly induce trait object lifetime defaults.
// issue: <https://github.com/rust-lang/rust/issues/115379>

//@ check-pass

mod own { // the trait object lifetime default comes from the own generics
    trait Outer {
        type Ty<'a, T: 'a + super::AbideBy<'a> + ?Sized>;
    }
    impl Outer for () {
        type Ty<'a, T: 'a + super::AbideBy<'a> + ?Sized> = ();
    }
    trait Inner {}

    // We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`.
    fn check<'r>(_: <() as Outer>::Ty<'r, dyn super::Inner>) {}
}

mod parent { // the trait object lifetime default comes from the parent generics
    trait Outer<'a> {
        type Ty<T: 'a + super::AbideBy<'a> + ?Sized>;
    }
    impl<'a> Outer<'a> for () {
        type Ty<T: 'a + super::AbideBy<'a> + ?Sized> = ();
    }

    // We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`.
    fn check<'r>(_: <() as Outer<'r>>::Ty<dyn super::Inner>) {}
}

#[rustfmt::skip]
mod complex {
    // We need to perform several delicate index calculations to map between the middle::ty and
    // the HIR representation of generic args. This is a smoke test.

    trait Outer<'_pad0, 'a, '_pad1, _Pad0> {
        type Ty<
            '_pad2, 'b, '_pad3, _Pad1,
            T: 'a + super::AbideBy<'a> + ?Sized,
            _Pad2,
            U: 'b + super::AbideBy<'b> + ?Sized,
        >;
    }
    impl<'a, _Pad0> Outer<'_, 'a, '_, _Pad0> for () {
        type Ty<
            '_pad2, 'b, '_pad3, _Pad1,
            T: 'a + super::AbideBy<'a> + ?Sized,
            _Pad2,
            U: 'b + super::AbideBy<'b> + ?Sized,
        > = ();
    }

    // We elaborate the 1st `dyn Inner` to `dyn Inner + 'r` and the 2nd one to `dyn Inner + 's`.
    fn g<'r, 's>(
        _: <() as Outer<'static, 'r, 'static, ()>>
            ::Ty<'static, 's, 'static, (), dyn super::Inner, (), dyn super::Inner>,
    ) {}
}

trait Inner {}

// We use this to test that a given trait object lifetime bound is
// *exactly equal* to a given lifetime (not longer, not shorter).
trait AbideBy<'a> {}
impl<'a> AbideBy<'a> for dyn Inner + 'a {}

fn main() {}
