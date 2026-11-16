// This test checks that outside of `core`/`alloc`/`std`, we don't link to incoherent method impl
// because there is no page to link to since primitive are only documented in `core` and `std`.

//@ aux-build: incoherent_impl_primitive.rs
//@ build-aux-docs
//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

extern crate incoherent_impl_primitive;

//@ has 'src/foo/incoherent_impl-primitive.rs.html'

// It should contain only the link to "first". Sadly, because of the limit, we cannot test
// if a link doesn't have an attribute so for now, we simply ensure that the number of `<a>`
// (including both line numbers and jump to def links) is superior by one to the number of
// line `<a>`. If so, we know there is only one jump to def link.
//@ count - '//pre/code/a' 7
//@ count - '//pre/code/a[@id]' 6
//@ has - '//pre/code/a[@href="{{channel}}/core/primitive.slice.html#method.first"]' 'first'

fn foo() {
    let x = ["a"].f();
    ["a"].first();
}
