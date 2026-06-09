// Ensure that structs with fields whose types contain diverging aliases are properly caught with
// both solvers.
// MCVE from https://github.com/rust-lang/trait-system-refactor-initiative/issues/139#issuecomment-2703127026.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

trait Trait {
    type Diverges<D: Trait>;
}

impl<T> Trait for T {
    type Diverges<D: Trait> = D::Diverges<D>;
}

struct Foo {
    field: Box<<u8 as Trait>::Diverges<u8>>,
    //[current]~^ ERROR: overflow evaluating the requirement `<u8 as Trait>::Diverges<u8> == _`
    //[next]~^^ ERROR: type mismatch resolving `<u8 as Trait>::Diverges<u8> normalizes-to _`
    //[next]~| ERROR: type mismatch resolving `<u8 as Trait>::Diverges<u8> normalizes-to _`
}

fn main() {}
