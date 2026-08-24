// Demonstrate that generic_const_args changes the behavior for dyn trait aliases
// with non-type associated consts: the associated const must be specified.

//@ revisions: no_gca gca
//@ compile-flags: -Znext-solver=globally
//@ [no_gca] check-pass

#![cfg_attr(gca, feature(generic_const_args, min_generic_const_args))]
#![cfg_attr(gca, expect(incomplete_features))]

type UnsatTraitBound0 = [str]; // `str: Sized` unsatisfied
type UnsatTraitBound1<T = Vec<str>> = T; // `str: Sized` unsatisfied
type UnsatOutlivesBound<'a> = &'static &'a (); // `'a: 'static` unsatisfied

type Diverging = [(); panic!()]; // `panic!()` diverging

type DynIncompat0 = dyn Sized; // `Sized` axiomatically dyn incompatible
// issue: <https://github.com/rust-lang/rust/issues/153731>
type DynIncompat1 = dyn HasAssocConst;
//[gca]~^ ERROR the value of the associated constant `N` in `HasAssocConst` must be specified

trait HasAssocConst {
    const N: usize;
}

fn main() {}
