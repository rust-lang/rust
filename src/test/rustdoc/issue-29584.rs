// aux-build:issue-29584.rs
// ignore-cross-compile

extern crate issue_29584;

// @has issue_29584/struct.Foo.html
// @!has - 'impl Bar for'
pub use issue_29584::Foo;
