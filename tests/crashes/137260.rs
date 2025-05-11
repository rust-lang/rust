//@ known-bug: #137260
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Iter<const N: usize = { 1 + true }> {}

fn needs_iter<const N: usize, T: Iter<N>>() {}

fn test() {
    needs_iter::<1, dyn Iter<()>>();
}
