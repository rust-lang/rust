// normalize-stderr-test "loaded from .*liballoc-.*.rlib" -> "loaded from SYSROOT/liballoc-*.rlib"
#![feature(lang_items)]

#[lang = "owned_box"]
struct Foo; //~ ERROR E0152

fn main() {
}
