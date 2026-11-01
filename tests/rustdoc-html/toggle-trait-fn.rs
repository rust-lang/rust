#![crate_name = "foo"]

// Trait methods with documentation should be wrapped in a <details> toggle with an appropriate
// summary. Trait methods with no documentation should not be wrapped.
//
//@ has foo/trait.Foo.html
//@ count - '//details[@class="toggle"]//summary//h4[@class="code-header"]' 1
//@ has - '//details[@class="toggle"]//summary//h4[@class="code-header"]' 'type Item'
//@ count -  '//details[@class="toggle method-toggle"]//summary//h4[@class="code-header"]' 2
//@ has -  '//details[@class="toggle method-toggle"]//summary//h4[@class="code-header"]' 'is_documented()'
//@ has -  '//details[@class="toggle method-toggle"]//*[@class="docblock"]' 'is_documented is documented'
//@ has -  '//details[@class="toggle method-toggle"]//summary//h4[@class="code-header"]' 'is_documented_optional()'
//@ has -  '//details[@class="toggle method-toggle"]//*[@class="docblock"]' 'is_documented_optional is documented'
pub trait Foo {
    /// is documented
    type Item;

    type Item2;

    fn not_documented();

    /// is_documented is documented
    fn is_documented();

    fn not_documented_optional() {}

    /// is_documented_optional is documented
    fn is_documented_optional() {}
}
