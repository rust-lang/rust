#![feature(decl_macro, use_extern_macros)]
pub struct Item;

pub macro foo() { Item }

pub macro bar() { Item }

#[macro_export]
macro_rules! baz {
    () => {
        Item
    }
}

pub macro qux1() { Item }

#[macro_export]
macro_rules! qux2 {
    () => {
        Item
    }
}

pub fn abc() -> Item {
    bar!()
}
