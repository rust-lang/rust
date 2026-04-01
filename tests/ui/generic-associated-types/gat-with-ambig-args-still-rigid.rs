//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#256. The ambiguous
// GAT arg check previously happened before checking whether the projection
// candidate actually applied in the new trait solver.
//
// This meant we didn't consider `T::Assoc<_>` to be a rigid alias, resulting
// in an inference failure.

pub trait Proj {
    type Assoc<T>;
}

trait Id {
    type This;
}
impl<T> Id for T {
    type This = T;
}

// This previously compiled as the "assumption would incompletely constrain GAT args"
// check happened in each individual assumption after the `DeepRejectCtxt` fast path.
fn with_fast_reject<T, U>(x: T::Assoc<u32>)
where
    T: Proj,
    U: Proj<Assoc<i32> = u32>,
{
    let _: T::Assoc<_> = x;
}

// This previously failed with ambiguity as that check did happen before we actually
// equated the goal with the assumption. Due to the alias in the where-clause
// we didn't fast-reject this candidate.
fn no_fast_reject<T, U>(x: T::Assoc<u32>)
where
    T: Proj,
    <U as Id>::This: Proj<Assoc<i32> = u32>,
{
    let _: T::Assoc<_> = x;
}

fn main() {}
