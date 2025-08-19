//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(const_trait_impl)]

#[const_trait]
trait Bar {}

trait Baz: const Bar {}

trait Foo {
    // Well-formedenss of `Baz` requires `<Self as Foo>::Bar: const Bar`.
    // Make sure we assemble a candidate for that via the item bounds.
    type Bar: Baz;
}

fn main() {}
