//@ edition: 2021
//@ aux-crate:issue_127011_a_2=issue-127011-a-2.rs

use issue_127011_a_2::Foo;

pub struct Bar;

impl Foo for Bar {
    fn foo() {}
}
