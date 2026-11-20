//@ check-pass
// Protection against a naive fix for #152607

//@ revisions: current next

//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Trait<T: Sized> {}

fn foo(_: &dyn Trait<[u32]>) {}

fn main() {}
