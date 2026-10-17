//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for a fairly subtle issue with how we
// generalize aliases referencing higher-ranked regions
// which previously caused unexpected ambiguity errors.
//
// The explanations in the test may end up being out of date
// in the future as we may refine our approach to generalization
// going forward.
//
// cc trait-system-refactor-initiative#108
trait Trait<'a> {
    type Assoc;
}

impl<'a> Trait<'a> for () {
    type Assoc = ();
}

fn foo<T: for<'a> Trait<'a>>(x: T) -> for<'a> fn(<T as Trait<'a>>::Assoc) {
    |_| ()
}

fn unconstrained<T>() -> T {
    todo!()
}

fn main() {
    // create `?x.0` in the root universe
    let mut x = unconstrained();

    // bump the current universe of the inference context
    let bump: for<'a, 'b> fn(&'a (), &'b ()) = |_, _| ();
    let _: for<'a> fn(&'a (), &'a ()) = bump;

    // create `?y.1` in a higher universe
    let mut y = Default::default();

    // relate `?x.0` with `for<'a> fn(<?y.1 as Trait<'a>>::Assoc)`
    // -> instantiate `?x.0` with `for<'a> fn(<?y_new.0 as Trait<'a>>::Assoc)`
    x = foo(y);

    // Constrain `?y.1` to `()`
    let _: () = y;

    // `AliasRelate(<?y_new.0 as Trait<'a>>::Assoc, <() as Trait<'a>>::Assoc)`
    // remains ambiguous unless we somehow constrain `?y_new.0` during
    // generalization to be equal to `?y.1`, which is exactly what we
    // did to fix this issue.
}
