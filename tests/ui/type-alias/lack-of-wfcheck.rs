// Demonstrate that we don't check the definition site of (eager) type aliases for well-formedness.
//
// Listed below are ill-formed type system entities which we don't reject since they appear inside
// the definition of (eager) type aliases. These type aliases are intentionally not referenced from
// anywhere to prevent the eagerly expanded / instantiated aliased types from getting wfchecked
// since that's not what we're testing here.

//@ check-pass

type UnsatTraitBound0 = [str]; // `str: Sized` unsatisfied
type UnsatTraitBound1<T = Vec<str>> = T; // `str: Sized` unsatisfied
type UnsatOutlivesBound<'a> = &'static &'a (); // `'a: 'static` unsatisfied

type Diverging = [(); panic!()]; // `panic!()` diverging

type DynIncompat0 = dyn Sized; // `Sized` axiomatically dyn incompatible
// * dyn incompatible due to GAT
// * `'a: 'static`, `String: Copy` and `[u8]: Sized` unsatisfied, `loop {}` diverging
type Several<'a> = dyn HasGenericAssocType<Type<'a, String, { loop {} }> = [u8]>;

trait HasGenericAssocType { type Type<'a: 'static, T: Copy, const N: usize>; }

fn main() {}
