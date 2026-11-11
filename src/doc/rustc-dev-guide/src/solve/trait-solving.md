# Trait solving (new)

This chapter describes how trait solving works with the new WIP solver located in
[`rustc_trait_selection/solve`][solve]. Feel free to also look at the docs for
[the current solver](../traits/resolution.md) and [the chalk solver](../traits/chalk.md).

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

[solve]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/index.html
[`Goal`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/canonical/ir/solve/struct.Goal.html
[`Predicate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Predicate.html
[`Candidate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_next_trait_solver/solve/assembly/struct.Candidate.html
[`CandidateSource`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/canonical/ir/solve/enum.CandidateSource.html
[`CanonicalResponse`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/solve/type.CanonicalResponse.html
