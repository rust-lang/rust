//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#279.
// The old solver treats unnormalized param env as rigid when normalizing param
// env. However, it eagerly evaluates const aliases in param env before doing
// full normalization.
// We previously treated unevaluated const aliases as rigid in the next solver.

trait Trait {
    type Assoc;
}

fn foo<T>()
where
    [T; 1 + 1]: Trait,
    // When normalizing this associated type, we couldn't prove the trait
    // predicate. Because the unnormalized trait clause is marked as rigid
    // and can't be used as a matching param env candidate.
    <[T; 2] as Trait>::Assoc: Copy,
{}

fn main() {}
