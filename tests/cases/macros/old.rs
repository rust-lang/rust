#![feature(decl_macro, use_extern_macros)]
pub struct Item;

pub macro bar() { Item }

fn abc() -> Item {
    bar!()
}
