#![crate_name = "foo"]

// Struct methods with documentation should be wrapped in a <details> toggle with an appropriate
// summary. Struct methods with no documentation should not be wrapped.
//
// @has foo/struct.Foo.html
// @has - '//details[@class="rustdoc-toggle method-toggle"]//summary//code' 'is_documented()'
// @has - '//details[@class="rustdoc-toggle method-toggle"]//*[@class="docblock"]' 'is_documented is documented'
// @!has - '//details[@class="rustdoc-toggle method-toggle"]//summary//code' 'not_documented()'
pub struct Foo {
}

impl Foo {
    pub fn not_documented() {}

    /// is_documented is documented
    pub fn is_documented() {}
}
