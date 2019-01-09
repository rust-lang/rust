// aux-build:issue-46727.rs

extern crate issue_46727;

// @has issue_46727/trait.Foo.html
// @has - '//code' 'impl<T> Foo for Bar<[T; 3]>'
pub use issue_46727::{Foo, Bar};
