#![feature(rustc_attrs)]

#![crate_name = "foo"]

pub use std::fs::File;

//@ has 'foo/primitive.i16.html' '//h2[@id="synthetic-implementations"]' 'Auto Trait Implementation'
#[rustc_doc_primitive = "i16"]
/// I love poneys!
mod prim {}
