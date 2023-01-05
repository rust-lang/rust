// aux-build:issue-61592.rs

extern crate foo;

#[doc = "bar"]
#[doc(inline)] //~ ERROR
#[doc = "baz"]
pub use foo::Foo as _;

fn main() {}
