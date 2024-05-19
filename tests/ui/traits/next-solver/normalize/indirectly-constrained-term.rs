//@ revisions: current next
//@[next] compile-flags: -Znext-solver=coherence
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// A regression test for `paperclip-core`. This previously failed to compile
// in the new solver.
//
// Behavior in old solver:
// We prove `Projection(<W<?0> as Unconstrained>::Assoc, ())`. This
// normalizes `<W<?0> as Unconstrained>::Assoc` to `?1` with nested goals
// `[Projection(<?0 as Unconstrained>::Assoc, ?1), Trait(?1: NoImpl)]`.
// We then unify `?1` with `()`. At this point `?1: NoImpl` does not hold,
// and we get an error.
//
// Previous behavior of the new solver:
// We prove `Projection(<W<?0> as Unconstrained>::Assoc, ())`. This normalizes
// `<W<?0> as Unconstrained>::Assoc` to `?1` and eagerly computes the nested
// goals `[Projection(<?0 as Unconstrained>::Assoc, ?1), Trait(?1: NoImpl)]`.
// These goals are both ambiguous. `NormalizesTo`` then returns `?1` as the
// normalized-to type. It discards the nested goals, forcing the certainty of
// the normalization to `Maybe`. Unifying `?1` with `()` succeeds¹. However,
// this is never propagated to the `?1: NoImpl` goal, as it only exists inside
// of the `NormalizesTo` goal. The normalized-to term always starts out as
// unconstrained.
//
// We fix this regression by returning the nested goals of `NormalizesTo` goals
// to the `AliasRelate`. This results in us checking `(): NoImpl`, same as the
// old solver.

struct W<T: ?Sized>(T);
trait NoImpl {}
trait Unconstrained {
    type Assoc;
}
impl<T: Unconstrained<Assoc = U>, U: NoImpl> Unconstrained for W<T> {
    type Assoc = U;
}


trait Overlap {}
impl<T: Unconstrained<Assoc = ()>> Overlap for T {}
impl<U> Overlap for W<U> {}

fn main() {}
