//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass

// Regression test from trait-system-refactor-initiative#27.

trait Trait<'t> {}
impl<'t> Trait<'t> for () {}

fn foo<'x, 'y>() -> impl Trait<'x> + Trait<'y> {}

fn impls_trait<'x, T: Trait<'x>>(_: T) {}

fn bar<'x, 'y>() {
    impls_trait::<'y, _>(foo::<'x, 'y>());
    //[next]~^ ERROR type annotations needed: cannot satisfy `impl Trait<'_> + Trait<'_>: Trait<'_>`
}

fn main() {}
