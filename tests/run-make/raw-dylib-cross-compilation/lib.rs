// tidy-alphabetical-start
#![feature(lang_items)]
#![feature(no_core)]
// tidy-alphabetical-end
#![no_std]
#![no_core]
#![crate_type = "lib"]

// This is needed because of #![no_core]:
#[lang = "sized"]
trait Sized {}

#[link(name = "extern_1", kind = "raw-dylib")]
extern "C" {
    fn extern_fn();
}

pub fn extern_fn_caller() {
    unsafe {
        extern_fn();
    }
}
