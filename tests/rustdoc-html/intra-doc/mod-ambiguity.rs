#![deny(rustdoc::broken_intra_doc_links)]


pub fn foo() {

}

pub mod foo {}
//@ has mod_ambiguity/type.A.html '//a/@href' 'foo/index.html'
/// Module is [`module@foo`]
pub struct A;


//@ has mod_ambiguity/type.B.html '//a/@href' 'fn.foo.html'
/// Function is [`fn@foo`]
pub struct B;
