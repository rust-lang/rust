//@ check-pass
//@ compile-flags: -Znext-solver

// Minimized example from `rustc_type_ir` that demonstrates a missing deep normalization
// in the new solver when computing the implies outlives bounds of an impl.

use std::marker::PhantomData;
use std::ops::Deref;

pub struct SearchGraph<D: Delegate, X = <D as Delegate>::Cx> {
    d: PhantomData<D>,
    x: PhantomData<X>,
}

pub trait Delegate {
    type Cx;
}

struct SearchGraphDelegate<D: SolverDelegate> {
    _marker: PhantomData<D>,
}

impl<D> Delegate for SearchGraphDelegate<D>
where
    D: SolverDelegate,
{
    type Cx = D::Interner;
}

pub trait SolverDelegate {
    type Interner;
}

struct EvalCtxt<'a, D, I>
where
    D: SolverDelegate<Interner = I>,
{
    search_graph: &'a SearchGraph<SearchGraphDelegate<D>>,
}

impl<'a, D, I> EvalCtxt<'a, D, <D as SolverDelegate>::Interner>
where
    D: SolverDelegate<Interner = I>
{}

fn main() {}
