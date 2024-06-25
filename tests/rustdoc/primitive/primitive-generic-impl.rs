#![feature(rustc_attrs)]
#![crate_name = "foo"]

//@ has foo/primitive.i32.html '//*[@id="impl-ToString-for-T"]//h3[@class="code-header"]' 'impl<T> ToString for T'

#[rustc_doc_primitive = "i32"]
/// Some useless docs, wouhou!
mod i32 {}
