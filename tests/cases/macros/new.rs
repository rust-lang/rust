#![feature(decl_macro, use_extern_macros)]
pub struct Item;

pub macro foo() { Item }

#[macro_export]
macro_rules! bar {
    () => {
        Item
    }
}

pub macro baz() { Item }

pub macro quux1() { Item }

#[macro_export]
macro_rules! quux2 {
    () => {
        Item
    }
}

pub fn abc() -> Item {
    bar!()
}
