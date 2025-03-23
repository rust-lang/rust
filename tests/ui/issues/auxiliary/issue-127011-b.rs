//@ edition: 2021
//@ aux-crate:issue_127011_a=issue-127011-a.rs

pub use issue_127011_a::Foo;

pub struct Bar;

impl Foo for Bar {
    fn foo() {}
}
