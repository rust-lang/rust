//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// When generalizing an alias in a bivariant context, we have to set
// `has_unconstrained_ty_var` as we may otherwise never check for
// well-formedness of the generalized type, causing us to error due
// to ambiguity.
trait Trait {
    type Assoc;
}

struct BivariantArg<I, T: Trait<Assoc = I>>(T);

fn generalize<T: Trait>(input: BivariantArg<T::Assoc, T>) {
    let _generalized = input;
}

pub fn main() {}
