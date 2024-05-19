//@ compile-flags: -Znext-solver
//@ check-pass

// With #119106 generalization now results in `AliasRelate` if the generalized
// alias contains an inference variable which is not nameable.
//
// We previously proved alias-relate after canonicalization, which does not keep track
// of universe indices, so all inference vars were nameable inside of `AliasRelate`.
//
// If we now have a rigid projection containing an unnameable inference variable,
// we should emit an alias-relate obligation, which constrains the type of `x` to
// an alias. This caused us to emit yet another equivalent alias-relate obligation
// when trying to instantiate the query result, resulting in overflow.
trait Trait<'a> {
    type Assoc: Default;
}

fn takes_alias<'a, T: Trait<'a>>(_: <T as Trait<'a>>::Assoc) {}

fn foo<T: for<'a> Trait<'a>>() {
    let x = Default::default();

    let _incr_universe: for<'a, 'b> fn(&'a (), &'b ()) =
        (|&(), &()| ()) as for<'a> fn(&'a (), &'a ());

    takes_alias::<T>(x);
}

fn main() {}
