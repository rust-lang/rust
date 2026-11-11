//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// This caused a regression in a crater run in #132325.
//
// The underlying issue is a really subtle implementation detail.
//
// When building the `param_env` for `Trait` we start out with its
// explicit predicates `Self: Trait` and `Self: for<'a> Super<'a, { 1 + 1 }>`.
//
// When normalizing the environment we also elaborate. This implicitly
// deduplicates its returned predicates. We currently first eagerly
// normalize constants in the unnormalized param env to avoid issues
// caused by our lack of deferred alias equality.
//
// So we actually elaborate `Self: Trait` and `Self: for<'a> Super<'a, 2>`,
// resulting in a third `Self: for<'a> Super<'a, { 1 + 1 }>` predicate which
// then gets normalized to  `Self: for<'a> Super<'a, 2>` at which point we
// do not deduplicate however. By failing to handle equal where-bounds in
// candidate selection, this caused ambiguity when checking that `Trait` is
// well-formed.
trait Super<'a, const N: usize> {}
trait Trait: for<'a> Super<'a, { 1 + 1 }> {}
fn main() {}
