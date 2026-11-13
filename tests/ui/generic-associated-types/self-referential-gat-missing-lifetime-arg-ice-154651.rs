//! Regression test for <https://github.com/rust-lang/rust/issues/154651>.
//!
//! A GAT defined in terms of itself with a missing lifetime argument used to ICE
//! with "cannot find `..` in param-env" under the new solver.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Processor {
    type Ctx<'a, const N: usize>;
}

pub struct Phasor;

impl Processor for Phasor {
    type Ctx<'a, const N: usize> = Self::Ctx<N>;
    //~^ ERROR associated type takes 1 lifetime argument but 0 lifetime arguments were supplied
}

fn main() {}
