// compile-flags: --crate-type lib --edition 2018

#![crate_name = "foo"]
#![feature(rustdoc_internals)]

// @has foo/primitive.tuple.html '//a[@class="primitive"]' 'tuple'
// @has - '//span[@class="in-band"]' 'Primitive Type tuple'
// @has - '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'
// @has - '//h2[@id="synthetic-implementations"]' 'Auto Trait Implementations'
// @has - '//div[@id="synthetic-implementations-list"]//h3' 'Send'
// @has - '//div[@id="synthetic-implementations-list"]//h3' 'Sync'
#[doc(primitive = "tuple")]
/// this is a test!
mod tuple_prim {}
