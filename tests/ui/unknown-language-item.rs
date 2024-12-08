#![allow(unused)]
#![feature(lang_items)]

#[lang = "foo"]
fn bar() -> ! {
//~^^ ERROR definition of an unknown lang item: `foo`
    loop {}
}

fn main() {}
