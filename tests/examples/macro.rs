#![feature(decl_macro, use_extern_macros)]

pub mod old {
    pub struct Item;

    pub macro bar() { Item }

    fn abc() -> Item {
        bar!()
    }
}

pub mod new {
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
}
