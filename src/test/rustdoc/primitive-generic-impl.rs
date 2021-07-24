#![crate_name = "foo"]

include!("primitive/primitive-generic-impl.rs");

// @has foo/primitive.i32.html '//div[@id="impl-ToString"]//h3[@class="code-header in-band"]' 'impl<T> ToString for T'
