//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// A test for an edge case of #160443. While we must not use the
// hidden type of opaques when computing the implied bounds for a function
// we should do so for nested bodies. This is necessary as otherwise
// normalizing their well-formedness requirements can fail.
//
// Closures are always checked for WF in their parent body, which can also
// reveal the hidden types of opaque types.

trait Trait {
    type Assoc;
}
impl Trait for () {
    type Assoc = ();
}

trait Func {
    type Output;
}
impl<F: FnOnce() -> R, R> Func for F {
    type Output = R;
}

struct RequiresWf<F>(F)
where
    F: Func,
    F::Output: Trait,
    <F::Output as Trait>::Assoc: 'static;

fn opaque() -> impl Sized {
    (|_| ())(RequiresWf(opaque));
    //[current]~^ ERROR the trait bound `impl Sized: Trait` is not satisfied
    //[current]~| ERROR the trait bound `impl Sized: Trait` is not satisfied
    //[current]~| ERROR the trait bound `impl Sized: Trait` is not satisfied
    //[current]~| ERROR the trait bound `impl Sized: Trait` is not satisfied
}

fn main() {}
