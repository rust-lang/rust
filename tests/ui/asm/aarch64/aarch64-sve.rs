//@ only-aarch64
//@ build-pass
//@ needs-asm-support

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]

// AArch64 test corresponding to arm64ec-sve.rs.

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

impl Copy for f64 {}

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

fn f(x: f64) {
    unsafe {
        asm!("", out("p0") _);
        asm!("", out("ffr") _);
    }
}
