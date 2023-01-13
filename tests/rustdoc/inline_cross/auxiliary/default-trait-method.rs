#![feature(specialization)]

#![crate_name = "foo"]

pub trait Item {
    fn foo();
    fn bar();
    fn baz() {}
}

pub struct Foo;

impl Item for Foo {
    default fn foo() {}
    fn bar() {}
}
