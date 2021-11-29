// @has issue_33054/impls/struct.Foo.html
// @has - '//h3[@class="code-header in-band"]' 'impl Foo'
// @has - '//h3[@class="code-header in-band"]' 'impl Bar for Foo'
// @count - '//*[@id="trait-implementations-list"]//*[@class="impl has-srclink"]' 1
// @count - '//*[@id="main-content"]/details/summary/*[@class="impl has-srclink"]' 1
// @has issue_33054/impls/bar/trait.Bar.html
// @has - '//h3[@class="code-header in-band"]' 'impl Bar for Foo'
// @count - '//*[@class="struct"]' 1
pub mod impls;

#[doc(inline)]
pub use impls as impls2;
