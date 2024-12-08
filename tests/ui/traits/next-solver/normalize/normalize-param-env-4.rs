//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] known-bug: #92505
//@[current] check-pass

trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    type Assoc = T;
}

fn impls_trait<T: Trait>() {}

fn foo<T>()
where
    <T as Trait>::Assoc: Trait,
{
    // Trying to use `<T as Trait>::Assoc: Trait` to prove `T: Trait`
    // requires normalizing `<T as Trait>::Assoc`. We do not normalize
    // using impl candidates if there's a where-bound for that trait.
    //
    // We therefore check whether `T: Trait` is proven by the environment.
    // For that we try to apply the `<T as Trait>::Assoc: Trait` candidate,
    // trying to normalize its self type results in overflow.
    //
    // In the old solver we eagerly normalize the environment, ignoring the
    // unnormalized `<T as Trait>::Assoc: Trait` where-bound when normalizing
    // `<T as Trait>::Asosc`
    impls_trait::<T>();
}

fn main() {}
