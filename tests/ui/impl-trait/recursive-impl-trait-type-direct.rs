//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] check-pass

#![allow(unconditional_recursion)]

// The new solver does not allow unconstrained opaque types and unlike
// the old solver, does not fall back to `()` here.
//
// See https://github.com/rust-lang/trait-system-refactor-initiative/issues/144.

fn test() -> impl Sized {
    //[next]~^ ERROR: type annotations needed
    test()
}

fn main() {}
