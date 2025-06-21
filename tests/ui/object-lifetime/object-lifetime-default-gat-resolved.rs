// Check that we correctly deduce object lifetime defaults inside resolved GAT *paths*.
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

    fn f<'r>(x: <() as Outer>::Ty<'r, dyn Inner + 'r>) { g(x) }
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

    fn f<'r>(x: <() as Outer<'r>>::Ty<dyn Inner + 'r>) { g(x) }
    // We deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`.
    fn g<'r>(_: <() as Outer<'r>>::Ty<dyn Inner>) {}
}

fn main() {}
