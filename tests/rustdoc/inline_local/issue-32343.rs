// @!has issue_32343/struct.Foo.html
// @has issue_32343/index.html
// @has - '//code' 'pub use foo::Foo'
// @!has - '//code/a' 'Foo'
#[doc(no_inline)]
pub use foo::Foo;

// @!has issue_32343/struct.Bar.html
// @has issue_32343/index.html
// @has - '//code' 'pub use foo::Bar'
// @has - '//code/a' 'Bar'
#[doc(no_inline)]
pub use foo::Bar;

mod foo {
    pub struct Foo;
    pub struct Bar;
}

pub mod bar {
    // @has issue_32343/bar/struct.Bar.html
    pub use ::foo::Bar;
}
