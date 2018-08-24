// @has issue_33054/impls/struct.Foo.html
// @has - '//code' 'impl Foo'
// @has - '//code' 'impl Bar for Foo'
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 1
// @count - '//*[@id="main"]/*[@class="impl"]' 1
// @has issue_33054/impls/bar/trait.Bar.html
// @has - '//code' 'impl Bar for Foo'
// @count - '//*[@class="struct"]' 1
pub mod impls;

#[doc(inline)]
pub use impls as impls2;
