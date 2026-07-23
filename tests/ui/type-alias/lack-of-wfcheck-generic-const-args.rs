// Demonstrate that generic_const_args changes the behavior for dyn trait aliases
// with non-type associated consts: the associated const must be specified.

//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, min_generic_const_args)]
#![expect(incomplete_features)]

type UnsatTraitBound0 = [str]; // `str: Sized` unsatisfied
type UnsatTraitBound1<T = Vec<str>> = T; // `str: Sized` unsatisfied
type UnsatOutlivesBound<'a> = &'static &'a (); // `'a: 'static` unsatisfied

type Diverging = [(); panic!()]; // `panic!()` diverging

type DynIncompat0 = dyn Sized; // `Sized` axiomatically dyn incompatible
// issue: <https://github.com/rust-lang/rust/issues/153731>
type DynIncompat1 = dyn HasAssocConst;
//~^ ERROR the value of the associated constant `N` in `HasAssocConst` must be specified

// * dyn incompatible due to GAT
// * `'a: 'static`, `String: Copy` and `[u8]: Sized` unsatisfied, `loop {}` diverging
type Several<'a> = dyn HasGenericAssocType<Type<'a, String, { loop {} }> = [u8]>;

trait HasAssocConst {
    const N: usize;
}
trait HasGenericAssocType {
    type Type<'a: 'static, T: Copy, const N: usize>;
}

fn main() {}
