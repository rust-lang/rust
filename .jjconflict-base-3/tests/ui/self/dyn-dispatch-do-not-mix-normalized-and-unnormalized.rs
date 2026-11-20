//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/138937>.

// Previously, we'd take the normalized param env's clauses which included
// `<PF as TraitC>::Value = i32`, which comes from the supertraits of `TraitD`
// after normalizing `<PF as TraitC>::Value = <PF as TraitD>::Scalar`. Since
// `normalize_param_env_or_error` ends up re-elaborating `PF: TraitD`, we'd
// end up with both versions of this predicate (normalized and unnormalized).
// Since these projections preds are not equal, we'd fail with ambiguity.

trait TraitB<T> {}

trait TraitC: TraitB<Self::Value> {
    type Value;
}

trait TraitD: TraitC<Value = Self::Scalar> {
    type Scalar;
}

trait TraitE {
    fn apply<PF: TraitD<Scalar = i32>>(&self);
}

fn main() {}
