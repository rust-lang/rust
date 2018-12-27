#![crate_name = "foo"]

include!("primitive/primitive-generic-impl.rs");

// @has foo/primitive.i32.html '//h3[@id="impl-ToString"]//code' 'impl<T> ToString for T'
