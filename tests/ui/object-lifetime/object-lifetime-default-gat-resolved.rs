// Check that we correctly deduce object lifetime defaults inside resolved GAT paths.
// issue: <https://github.com/rust-lang/rust/issues/115379>

//@ check-pass

mod own { // the object lifetime default comes from the own generics
    trait Outer {
        type Ty<'a, T: ?Sized + 'a>;
    }
    impl Outer for () {
        type Ty<'a, T: ?Sized + 'a> = &'a T;
    }
    trait Inner {}

    fn f<'r>(x: <() as Outer>::Ty<'r, dyn Inner + 'r>) { /*check*/ g(x) }
    // We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`.
    fn g<'r>(_: <() as Outer>::Ty<'r, dyn Inner>) {}
}

mod parent { // the object lifetime default comes from the parent generics
    trait Outer<'a> {
        type Ty<T: ?Sized + 'a>;
    }
    impl<'a> Outer<'a> for () {
        type Ty<T: ?Sized + 'a> = &'a T;
    }
    trait Inner {}

    fn f<'r>(x: <() as Outer<'r>>::Ty<dyn Inner + 'r>) { /*check*/ g(x) }
    // We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`.
    fn g<'r>(_: <() as Outer<'r>>::Ty<dyn Inner>) {}
}

mod complex {
    // We need to perform several delicate index calculations to map between the middle::ty and
    // the HIR representation of generic args. This is a smoke test.

    trait Outer<'_pad0, 'a, '_pad1, _Pad0> {
        type Ty<'_pad2, 'b, '_pad3, _Pad1, T: ?Sized + 'a, _Pad2, U: ?Sized + 'b>;
    }
    impl<'a, _Pad0> Outer<'_, 'a, '_, _Pad0> for () {
        type Ty<'_pad2, 'b, '_pad3, _Pad1, T: ?Sized + 'a, _Pad2, U: ?Sized + 'b> = (&'a T, &'b U);
    }
    trait Inner {}

    fn f<'r, 's>(
        x: <() as Outer<'static, 'r, 'static, ()>>
            ::Ty<'static, 's, 'static, (), dyn Inner + 'r, (), dyn Inner + 's>,
    ) {
        /*check*/ g(x)
    }
    // We elaborate the 1st `dyn Inner` to `dyn Inner + 'r` and the 2nd one to `dyn Inner + 's`.
    fn g<'r, 's>(
        _: <() as Outer<'static, 'r, 'static, ()>>
            ::Ty<'static, 's, 'static, (), dyn Inner, (), dyn Inner>,
    ) {}
}

fn main() {}
