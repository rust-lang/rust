//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/248>
// Test whether we can call methods on not-yet defined nested projections on opaques such as
// `<<{opaque}> as Baz>::Assoc as Bar>::Assoc`.

#![allow(warnings)]

trait Foo {
    fn foo(&self) {}
}

trait Bar {
    type Assoc: Foo;

    fn bar(&self) -> Self::Assoc {
        loop {}
    }
}

trait Baz {
    type Assoc: Bar;

    fn baz(&self) -> Self::Assoc {
        loop {}
    }
}

impl Foo for () {}

impl Bar for () {
    type Assoc = ();
}

impl Baz for () {
    type Assoc = ();
}

fn heck() -> impl Baz {
    heck().baz().bar().foo()
}

fn main() {}
