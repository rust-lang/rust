// This checks that if the item is not directly `doc(hidden)` but instead a parent
// is, we don't generate a link as it would lead to non-existing files.

//@ compile-flags: -Zunstable-options --generate-link-to-definition
//@ aux-build:hidden-links.rs
//@ build-aux-docs

#![crate_name = "foo"]

extern crate hidden_links;

//@ has 'src/foo/hidden-links.rs.html'

fn foo() {
    // In here, only `Bar` should be linked as `Foo` has a `doc(hidden)` parent and
    // is not reexported. So there should be only one link.

    // Because our XPath handler is very bad, I can't look for `not(@id)` so instead,
    // we count how many `<a>` we have in total (line numbers are `<a>`).
    //@ count - '//pre/code/a' 7
    // Then we count the number of lines. The diff should be one (no way to test that either..).
    //@ count - '//pre/code/a[@id]' 6
    //@ has - '//pre/code/a[@href="../../hidden_links/struct.Bar.html"]' 'Bar'
    let x = hidden_links::error::Foo;
    let x = hidden_links::Bar;
}
