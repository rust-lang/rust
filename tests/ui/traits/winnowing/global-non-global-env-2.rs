//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// A regression test for an edge case of candidate selection
// in the old trait solver, see #132325 for more details. Unlike
// the first test, this one has two impl candidates.

trait Trait<T> {}
impl Trait<u32> for () {}
impl Trait<u64> for () {}

fn impls_trait<T: Trait<U>, U>(_: T) -> U { todo!() }
fn foo<T>() -> u32
where
    (): Trait<u32>,
    (): Trait<T>,
{
    impls_trait(())
}

fn main() {}
