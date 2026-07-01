//@ check-pass
//@ compile-flags: -Znext-solver
#![allow(self_type_conversion)]

struct NewSolver;
struct OldSolver;

fn foo<T>()
where
    T: Iterator<Item = NewSolver>,
    OldSolver: Into<T::Item>,
{
    let x: OldSolver = OldSolver.into();
}

fn main() {}
