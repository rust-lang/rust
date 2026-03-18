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
// issue: <https://github.com/rust-lang/rust/issues/153731>
type DynIncompat1 = dyn HasAssocConst; // dyn incompatible due to (non-type-level) assoc const

// * dyn incompatible due to GAT
// * `'a: 'static`, `String: Copy` and `[u8]: Sized` unsatisfied, `loop {}` diverging
type Several<'a> = dyn HasGenericAssocType<Type<'a, String, { loop {} }> = [u8]>;

trait HasAssocConst { const N: usize; }
trait HasGenericAssocType { type Type<'a: 'static, T: Copy, const N: usize>; }

fn main() {}
