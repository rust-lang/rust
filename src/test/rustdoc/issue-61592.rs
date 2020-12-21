pub mod foo {
    pub struct Foo;
}

// @has issue_61592/index.html
// @has - '//*[@href="#reexports"]' 'Re-exports'
pub use foo::Foo as _;
