// Demonstrate that we don't check the def site of (unchecked) type aliases for well-formedness.
//
// Listed below are ill-formed type system entities which we don't reject since they appear inside
// the definition of (unchecked) type aliases. These type aliases are intentionally not referenced
// from anywhere to prevent the eagerly expanded / instantiated aliased types from getting wfchecked
// since that's not what we're testing here.

//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

type UnsatTraitBound0 = [str]; // `str: Sized` unsatisfied
type UnsatTraitBound1<T = Vec<str>> = T; // `str: Sized` unsatisfied
type UnsatOutlivesBound<'a> = &'static &'a (); // `'a: 'static` unsatisfied

type Diverging = [(); panic!()]; // `panic!()` diverging

// * `'a: 'static`, `String: Copy` and `[u8]: Sized` unsatisfied, `loop {}` diverging
#[expect(unused_associated_type_bounds)]
type Several<'a> = dyn Trait<Type<'a, String, { loop {} }> = [u8]>;

trait Trait {
    type Type<'a: 'static, T: Copy, const N: usize>
    where
        Self: Sized;
}

fn main() {}
