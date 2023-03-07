#![crate_name = "foo"]

#![feature(rustdoc_internals)]

// @has foo/index.html '//h2[@id="primitives"]' 'Primitive Types'
// @has foo/index.html '//a[@href="primitive.i32.html"]' 'i32'
// @has foo/index.html '//div[@class="sidebar-elems"]//li/a' 'Primitive Types'
// @has foo/index.html '//div[@class="sidebar-elems"]//li/a/@href' '#primitives'
// @has foo/primitive.i32.html '//a[@class="primitive"]' 'i32'
// @has foo/primitive.i32.html '//h1' 'Primitive Type i32'
// @has foo/primitive.i32.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'
// @has foo/index.html '//a/@href' '../foo/index.html'
// @!has foo/index.html '//span' '🔒'
#[doc(primitive = "i32")]
/// this is a test!
mod i32{}

// @has foo/primitive.bool.html '//section[@id="main-content"]//div[@class="docblock"]//p' 'hello'
#[doc(primitive = "bool")]
/// hello
mod bool {}
