#![crate_name = "foo"]

include!("primitive/primitive-generic-impl.rs");

// @has foo/primitive.i32.html '//div[@id="impl-ToString"]//h3' 'impl<T> ToString for T'
