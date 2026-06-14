//@ check-pass

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Tr<X>: 'static {}

struct S<'a, X>(&'a dyn Tr<X>);

fn main() {}
