#![feature(decl_macro, use_extern_macros)]
pub struct Item;

#[macro_export]
macro_rules! bar {
    () => {
        Item
    }
}

#[allow(dead_code)]
fn abc() -> Item {
    bar!()
}
