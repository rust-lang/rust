// This checks that if the item is not directly `doc(hidden)` but instead a parent
// is, we don't generate a link as it would lead to non-existing files.

//@ compile-flags: -Zunstable-options --generate-link-to-definition
//@ aux-build:hidden-links.rs
//@ build-aux-docs

#![crate_name = "foo"]

extern crate hidden_links;

//@ has 'src/foo/hidden-links.rs.html'

// Because our XPath handler is very bad, I can't look for `not(@id)` so instead,
// we count how many `<a>` we have in total (line numbers are `<a>`).
//@ count - '//pre/code/a' 21
// Then we count the `<a>` of lines. The diff should be the number of non-line `<a>` (no
// better way to test that either..).
//@ count - '//pre/code/a[@id]' 13

//@ has - '//pre/code/a[@href="../../hidden_links/trait.PubTrait.html"]' 'PubTrait'
use hidden_links::{PubTrait, error::HiddenTrait};

fn foo() {
    // In here, only `Bar` should be linked as `Foo` has a `doc(hidden)` parent and
    // is not reexported. So there should be only one link.

    // This one should not generate any link.
    let x = hidden_links::error::Foo;
    //@ has - '//pre/code/a[@href="../../hidden_links/struct.Bar.html"]' 'Bar'
    let y = hidden_links::Bar;

    // `x` and `y` generate a link every time...
    //@ count - '//pre/code/a[@href="#29"]' 2
    //@ count - '//pre/code/a[@href="#31"]' 2
    // FIXME(GuillaumeGomez): This is a bug, it should link the trait method on the item's page,
    // not on the trait's (and so `x.public()`) should not link.
    //@ count - '//pre/code/a[@href="../../hidden_links/trait.PubTrait.html#method.public"]' 2
    x.hidden();
    y.hidden();
    x.public();
    y.public();
}
