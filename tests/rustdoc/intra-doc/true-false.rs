#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "foo"]

//@ has foo/index.html
//@ has - '//*[@id="main-content"]//a[@href="{{channel}}/std/primitive.bool.html"]' 'true'
//@ has - '//*[@id="main-content"]//a[@href="{{channel}}/std/primitive.bool.html"]' 'false'

//! A `bool` is either [`true`] or [`false`].
