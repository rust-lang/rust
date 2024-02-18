//@ run-pass

// rpass test for reservation impls. Not 100% required because `From` uses them,
// but still.

//@ revisions: old next
//@[next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]

use std::mem;

trait MyTrait<S> {
    fn foo(&self, s: S) -> usize;
}

#[rustc_reservation_impl = "foo"]
impl<T> MyTrait<u64> for T {
    fn foo(&self, _x: u64) -> usize { 0 }
}

// reservation impls don't create coherence conflicts, even with
// non-chain overlap.
impl<S> MyTrait<S> for u32 {
    fn foo(&self, _x: S) -> usize { mem::size_of::<S>() }
}

fn main() {
    // ...and the non-reservation impl gets picked.XS
    assert_eq!(0u32.foo(0u64), mem::size_of::<u64>());
}
