//@ normalize-stderr: "loaded from .*liballoc-.*.rmeta" -> "loaded from SYSROOT/liballoc-*.rmeta"
#![feature(lang_items)]

#[lang = "owned_box"]
struct Foo<T>(T); //~ ERROR E0152

fn main() {
}
