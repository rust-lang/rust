// Regression test for trait-system-refactor-initiative#114.
//
// We previously treated the cycle when trying to use the
// `<R as DimMin<C>>::Output: DimMin` where-bound when
// normalizing `<R as DimMin<C>>::Output` as ambiguous, causing
// this to error.

//@ check-pass
//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver

pub trait DimMin<D> {
    type Output;
}
pub fn repro<R: DimMin<C>, C>()
where
    <R as DimMin<C>>::Output: DimMin<C, Output = <R as DimMin<C>>::Output>,
{
}

fn main() {}
