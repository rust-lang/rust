#![crate_name = "foo"]

#![feature(rustc_attrs)]

//@ has foo/primitive.i32.html '//h1' 'Primitive Type i32'
//@ has foo/index.html '//a/@href' '../foo/index.html'
#[rustc_doc_primitive = "i32"]
mod i32 {}

//@ has foo/struct.Node.html '//a/@href' 'primitive.i32.html'
pub struct Node;

impl Node {
    pub fn edge(&self) -> i32 { 0 }
}

//@ !has foo/type.Alias.html '//a/@href' 'primitive.i32.html'
//@ hasraw 'type.impl/foo/struct.Node.js' 'href=\"foo/primitive.i32.html\"'
pub type Alias = Node;
