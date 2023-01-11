#![feature(lang_items)]

#[lang = "cookie"]
fn cookie() -> ! {
//~^^ ERROR definition of an unknown language item: `cookie` [E0522]
    loop {}
}

fn main() {}
