#![feature(raw_dylib)]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

// This is needed because of #![no_core]:
#[lang = "sized"]
trait Sized {}

#[link(name = "extern_1", kind = "raw-dylib")]
extern {
    fn extern_fn();
}

pub fn extern_fn_caller() {
    unsafe {
        extern_fn();
    }
}
