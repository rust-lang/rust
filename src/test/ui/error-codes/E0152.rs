#![feature(lang_items)]

#[lang = "panic_impl"]
struct Foo; //~ ERROR E0152

fn main() {
}
