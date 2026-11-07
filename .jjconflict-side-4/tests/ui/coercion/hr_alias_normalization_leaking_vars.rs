// Regression test for #132765
//
// We have two function parameters with types:
// - `&?0`
// - `Box<for<'a> fn(<?0 as Trait<'a>>::Item)>`
//
// As the alias in the second parameter has a `?0` it is an ambig
// alias, and as it references bound vars it can't be normalized to
// an infer var.
//
// When checking function arguments we try to coerce both:
// - `&()` to `&?0`
// - `FnDef(f)` to `Box<for<'a> fn(<?0 as Trait<'a>>::Item)>`
//
// The first coercion infers `?0=()`. Previously when handling
// the second coercion we wound *re-normalize* the alias, which
// now that `?0` has been inferred allowed us to determine this
// alias is not wellformed and normalize it to some infer var `?1`.
//
// We would then see that `FnDef(f)` can't be coerced to `Box<fn(?1)>`
// and return a `TypeError` referencing this new variable `?1`. This
// then caused ICEs as diagnostics would encounter inferences variables
// from the result of normalization inside of the probe used be coercion.


trait LendingIterator {
    type Item<'q>;
    fn for_each(&self, _f: Box<fn(Self::Item<'_>)>) {}
}

fn f(_: ()) {}

fn main() {
    LendingIterator::for_each(&(), f);
    //~^ ERROR: the trait bound `(): LendingIterator` is not satisfied
    //~| ERROR: the trait bound `(): LendingIterator` is not satisfied
    //~| ERROR: mismatched types
}
