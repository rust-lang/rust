//@ compile-flags: -Znext-solver=coherence
//@ check-pass
//@ aux-build:coherence-unknowable-does-not-eagerly-normalize-dep.rs
//
// Refer issue: https://github.com/rust-lang/rust/issues/157407
//
// The reason this regression happened is because `<PgRow as Row>::Database`
// does not get normalized to `()`, but instead
// `ActivityType: Decode<<PgRow as Row>::Database>` gets treated like
// `ActivityType: Decode<?unconstrained>`, staying ambiguous.
// Now it is considered that a downstream crate may provide such an impl, even
// though `<PgRow as Row>::Database` can only normalize to a local or upstream
// type, thus causing this.

extern crate coherence_unknowable_does_not_eagerly_normalize_dep as dep;

use dep::{PgRow, Row};

trait FromRow {}
trait Decode<T> {}

struct ActivityType;

impl FromRow for PgRow {}

impl<R: Row> FromRow for R where ActivityType: Decode<R::Database> {}

fn main() {}
