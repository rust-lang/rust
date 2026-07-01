// Regression test for #152607: directly tests that a `dyn Trait` type whose
// associated type violates a supertrait bound is rejected.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Super {
    type Assoc;
}

trait Sub: Super<Assoc: Copy> {}

fn use_dyn(_: &dyn Sub<Assoc = String>) {}
//~^ ERROR

fn main() {}
