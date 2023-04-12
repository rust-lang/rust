# Trait solving (new)

This chapter describes how trait solving works with the new WIP solver located in
[`rustc_trait_selection/solve`][solve]. Feel free to also look at the docs for
[the current solver](../traits/resolution.md) and [the chalk solver](../traits/chalk.md)
can be found separately.

## Core concepts

The goal of the trait system is to check whether a given trait bound is satisfied.
Most notably when typechecking the body of - potentially generic - functions.
For example:

```rust
fn uses_vec_clone<T: Clone>(x: Vec<T>) -> (Vec<T>, Vec<T>) {
    (x.clone(), x)
}
```
Here the call to `x.clone()` requires us to prove that `Vec<T>` implements `Clone` given
the assumption that `T: Clone` is true. We can assume `T: Clone` as that will be proven by
callers of this function.

The concept of "prove the `Vec<T>: Clone` with the assumption `T: Clone`" is called a [`Goal`].
Both `Vec<T>: Clone` and `T: Clone` are represented using [`Predicate`]. There are other
predicates, most notably equality bounds on associated items: `<Vec<T> as IntoIterator>::Item == T`.
See the `PredicateKind` enum for an exhaustive list. A `Goal` is represented as the `predicate` we
have to prove and the `param_env` in which this predicate has to hold.

We prove goals by checking whether each possible [`Candidate`] applies for the given goal by
recursively proving its nested goals. For a list of possible candidates with examples, look at
[`CandidateSource`]. The most important candidates are `Impl` candidates, i.e. trait implementations
written by the user, and `ParamEnv` candidates, i.e. assumptions in our current environment.

Looking at the above example, to prove `Vec<T>: Clone` we first use
`impl<T: Clone> Clone for Vec<T>`. To use this impl we have to prove the nested
goal that `T: Clone` holds. This can use the assumption `T: Clone` from the `ParamEnv`
which does not have any nested goals. Therefore `Vec<T>: Clone` holds.

The trait solver can either return success, ambiguity or an error as a [`CanonicalResponse`].
For success and ambiguity it also returns constraints inference and region constraints.

## Requirements

Before we dive into the new solver lets first take the time to go through all of our requirements
on the trait system. We can then use these to guide our design later on.

TODO: elaborate on these rules and get more precise about their meaning.
Also add issues where each of these rules have been broken in the past
(or still are).

### 1. The trait solver has to be *sound*

This means that we must never return *success* for goals for which no `impl` exists. That would
simply be unsound by assuming a trait is implemented even though it is not. When using predicates
from the `where`-bounds, the `impl` will be proved by the user of the item.

### 2. If type checker solves generic goal concrete instantiations of that goal have the same result

Pretty much: If we successfully typecheck a generic function concrete instantiations
of that function should also typeck. We should not get errors post-monomorphization.
We can however get overflow as in the following snippet:

```rust
fn foo<T: Trait>(x: )
```

### 3. Trait goals in empty environments are proven by a unique impl

If a trait goal holds with an empty environment, there is a unique `impl`,
either user-defined or builtin, which is used to prove that goal.

This is necessary for codegen to select a unique method.
An exception here are *marker traits* which are allowed to overlap.

### 4. Normalization in empty environments results in a unique type

Normalization for alias types/consts has a unique result. Otherwise we can easily implement
transmute in safe code. Given the following function, we have to make sure that the input and
output types always get normalized to the same concrete type.
```rust
fn foo<T: Trait>(
    x: <T as Trait>::Assoc
) -> <T as Trait>::Assoc {
    x
}
```

### 5. During coherence trait solving has to be complete

During coherence we never return *error* for goals which can be proven. This allows overlapping
impls which would break rule 3.

### 6. Trait solving must be (free) lifetime agnostic

Trait solving during codegen should have the same result as during typeck. As we erase
all free regions during codegen we must not rely on them during typeck. A noteworthy example
is special behavior for `'static`.

We also have to be careful with relying on equality of regions in the trait solver.
This is fine for codegen, as we treat all erased regions are equal. We can however
lose equality information from HIR to MIR typeck.

### 7. Removing ambiguity makes strictly more things compile

We *should* not rely on ambiguity for things to compile.
Not doing that will cause future improvements to be breaking changes.

### 8. semantic equality implies structural equality

Two types being equal in the type system must mean that they have the same `TypeId`.


[solve]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/index.html
[`Goal`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/struct.Goal.html
[`Predicate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Predicate.html
[`Candidate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/assembly/struct.Candidate.html
[`CandidateSource`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/assembly/enum.CandidateSource.html
[`CanonicalResponse`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/type.CanonicalResponse.html
