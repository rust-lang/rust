#![feature(lang_items)]

#[lang = "arc"]
struct Foo; //~ ERROR E0152

fn main() {
}
