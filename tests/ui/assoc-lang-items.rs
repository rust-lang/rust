#![feature(lang_items)]

trait Foo {
    #[lang = "dummy_lang_item_1"] //~ ERROR definition
    fn foo() {}

    #[lang = "dummy_lang_item_2"] //~ ERROR definition
    fn bar();

    #[lang = "dummy_lang_item_3"] //~ ERROR definition
    type MyType;
}

struct Bar;

impl Bar {
    #[lang = "dummy_lang_item_4"] //~ ERROR definition
    fn test() {}
}

fn main() {}
