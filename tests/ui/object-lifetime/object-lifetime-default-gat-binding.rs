// Ideally, GAT args in bindings would induce ambient object lifetime defaults.
//
// However, since the current implementation can't handle this we instead conservatively and hackily
// treat the ambient object lifetime default as indeterminate if any lifetime arguments are passed
// to the trait ref (or the GAT) thus rejecting any hidden object lifetime bounds.
// This way, we can still implement the desired behavior in the future.

mod own { // the object lifetime default comes from the own generics
    trait Outer {
        type Ty<'a, T: ?Sized + 'a>;
    }

    trait Inner {}

    fn f<'r>(x: impl Outer<Ty<'r, dyn Inner + 'r> = ()>) { /*check*/ g(x) }
    // FIXME: Ideally, we'd elaborate `dyn Inner` to `dyn Inner + 'r` instead of rejecting it.
    fn g<'r>(_: impl Outer<Ty<'r, dyn Inner> = ()>) {}
    //~^ ERROR please supply an explicit bound
}

mod parent { // the object lifetime default comes from the parent generics
    trait Outer<'a> {
        type Ty<T: ?Sized + 'a>;
    }

    trait Inner {}

    fn f<'r>(x: impl Outer<'r, Ty<dyn Inner + 'r> = ()>) { /*check*/ g(x) }
    // FIXME: Ideally, we'd elaborate `dyn Inner` to `dyn Inner + 'r` instead of rejecting it.
    fn g<'r>(_: impl Outer<'r, Ty<dyn Inner> = ()>) {}
    //~^ ERROR please supply an explicit bound
}

fn main() {}
