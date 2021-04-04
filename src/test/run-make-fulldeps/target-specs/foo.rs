#![feature(lang_items, no_core, auto_traits)]
#![no_core]

#[lang = "copy"]
trait Copy {}

#[lang = "sized"]
trait Sized {}

#[lang = "freeze"]
auto trait Freeze {}

#[lang = "start"]
fn start(_main: *const u8, _argc: isize, _argv: *const *const u8) -> isize {
    0
}

extern "C" {
    fn _foo() -> [u8; 16];
}

fn _main() {
    let _a = unsafe { _foo() };
}
