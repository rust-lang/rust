// Check that we handle evaluating `wf` predicates correctly.

//@ check-pass

struct X<T: B>(T)
where
    T::V: Clone;

fn hide<T>(t: T) -> impl Sized {
    t
}

trait A {
    type U;
}

impl<T> A for T {
    type U = T;
}

trait B {
    type V;
}

impl<S: A<U = T>, T> B for S {
    type V = T;
}

fn main() {
    // Evaluating `typeof(x): Sized` requires
    //
    // - `wf(typeof(x))` because we use a projection candidate.
    // - `<i32 as B>::V: Clone` because that's a bound on the trait.
    // - `<i32 as B>::V` normalizes to `?1t` where `<i32 as A>::U == ?1t`
    //
    // This all works if we evaluate `<i32 as A>::U == ?1t` before
    // `<i32 as B>::V`, but we previously had the opposite order.
    let x = hide(X(0));
}
