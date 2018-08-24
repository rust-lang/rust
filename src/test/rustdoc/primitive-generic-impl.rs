#![crate_name = "foo"]

// we need to reexport something from libstd so that `all_trait_implementations` is called.
pub use std::string::String;

include!("primitive/primitive-generic-impl.rs");

// @has foo/primitive.i32.html '//h3[@id="impl-ToString"]//code' 'impl<T> ToString for T'
