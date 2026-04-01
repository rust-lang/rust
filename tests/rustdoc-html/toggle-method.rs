#![crate_name = "foo"]

// Struct methods with documentation should be wrapped in a <details> toggle with an appropriate
// summary. Struct methods with no documentation should not be wrapped.
//
//@ has foo/struct.Foo.html
//@ has - '//details[@class="toggle method-toggle"]//summary//h4[@class="code-header"]' 'is_documented()'
//@ has - '//details[@class="toggle method-toggle"]//*[@class="docblock"]' 'is_documented is documented'
//@ !has - '//details[@class="toggle method-toggle"]//summary//h4[@class="code-header"]' 'not_documented()'
pub struct Foo {
}

impl Foo {
    pub fn not_documented() {}

    /// is_documented is documented
    pub fn is_documented() {}
}
