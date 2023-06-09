#![allow(unused)]
#![feature(lang_items)]

#[lang = "foo"]
fn bar() -> ! {
//~^^ ERROR definition of an unknown language item: `foo`
    loop {}
}

fn main() {}
