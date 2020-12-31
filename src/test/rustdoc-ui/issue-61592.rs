// aux-build:issue-61592.rs

extern crate foo;

#[doc(inline)] //~ ERROR
pub use foo::Foo as _;

fn main() {}
