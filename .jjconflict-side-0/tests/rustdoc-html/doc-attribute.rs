// Test checking the `#[doc(attribute = "...")]` attribute.

#![crate_name = "foo"]

#![feature(rustdoc_internals)]

//@ has foo/index.html '//h2[@id="attribute-docs"]' 'Attributes'
//@ has foo/index.html '//a[@href="attribute.no_mangle.html"]' 'no_mangle'
//@ has foo/index.html '//div[@class="sidebar-elems"]//li/a' 'Attributes'
//@ has foo/index.html '//div[@class="sidebar-elems"]//li/a/@href' '#attribute-docs'
//@ has foo/attribute.no_mangle.html '//h1' 'Attribute no_mangle'
//@ has foo/attribute.no_mangle.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'
//@ has foo/index.html '//a/@href' '../foo/index.html'
//@ !has foo/index.html '//span' '🔒'
#[doc(attribute = "no_mangle")]
/// this is a test!
const _: () = ();

//@ has foo/attribute.repr.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'hello'
#[doc(attribute = "repr")]
/// hello
const _: () = ();
