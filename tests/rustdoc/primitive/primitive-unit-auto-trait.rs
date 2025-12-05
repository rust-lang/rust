//@ compile-flags: --crate-type lib
//@ edition: 2018

#![crate_name = "foo"]
#![feature(rustc_attrs)]

//@ has foo/primitive.unit.html '//h1' 'Primitive Type unit'
//@ has - '//section[@id="main-content"]//div[@class="docblock"]//p' 'this is a test!'
//@ has - '//h2[@id="synthetic-implementations"]' 'Auto Trait Implementations'
//@ has - '//div[@id="synthetic-implementations-list"]//h3' 'impl Send for ()'
//@ has - '//div[@id="synthetic-implementations-list"]//h3' 'impl Sync for ()'
#[rustc_doc_primitive = "unit"]
/// this is a test!
mod unit_prim {}
