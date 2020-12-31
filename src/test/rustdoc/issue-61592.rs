// aux-build:issue-61592.rs

extern crate foo;

// @has issue_61592/index.html
// @has - '//a[@href="#reexports"]' 'Re-exports'
// @has - '//code' 'pub use foo::FooTrait as _;'
// @!has - '//a[@href="trait._.html"]'
pub use foo::FooTrait as _;

// @has issue_61592/index.html
// @has - '//a[@href="#reexports"]' 'Re-exports'
// @has - '//code' 'pub use foo::FooStruct as _;'
// @!has - '//a[@href="struct._.html"]'
pub use foo::FooStruct as _;
