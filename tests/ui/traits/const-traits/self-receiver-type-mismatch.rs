//! Regression test for https://github.com/rust-lang/rust/issues/112623.
//! The const evaluator used to ICE with an assertion failure on a size mismatch
//! when a trait impl changed the `self` receiver type from by-value to by-reference.

#![feature(const_trait_impl)]

const trait Func {
    fn trigger(self) -> usize;
}

struct Cls;

impl const Func for Cls {
    fn trigger(&self, a: usize) -> usize {
        //~^ ERROR method `trigger` has 2 parameters but the declaration in trait `Func::trigger` has 1
        0
    }
}

enum Bug<T = [(); Cls.trigger()]> {
    V(T),
}

fn main() {}
