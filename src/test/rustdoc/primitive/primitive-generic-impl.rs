#![feature(rustdoc_internals)]

#![crate_name = "foo"]

// @has foo/primitive.i32.html '//div[@id="impl-ToString"]//h3[@class="code-header in-band"]' 'impl<T> ToString for T'

#[doc(primitive = "i32")]
/// Some useless docs, wouhou!
mod i32 {}
