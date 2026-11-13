//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// A regression test for an edge case of candidate selection
// in the old trait solver, see #132325 for more details. Unlike
// the second test, the where-bounds are in a different order.

trait Trait<T> {}
impl Trait<u32> for () {}
impl Trait<u64> for () {}

fn impls_trait<T: Trait<U>, U>(_: T) -> U { todo!() }
fn foo<T>() -> u32
where
    (): Trait<T>,
    (): Trait<u32>,
{
    impls_trait(())
}

fn main() {}
