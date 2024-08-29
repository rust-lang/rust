//@ build-pass
//@ compile-flags: --target=armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
#![feature(no_core, rustc_attrs, decl_macro, lang_items)]
#![crate_type = "rlib"]
#![no_std]
#![no_core]

// We accidentally classified "d0"..="d15" as dregs, even though they are in dreg_low16,
// and thus didn't compile them on platforms with only 16 dregs.
// Highlighted in https://github.com/rust-lang/rust/issues/126797

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

impl Copy for f64 {}

#[rustc_builtin_macro]
pub macro asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}


fn f(x: f64) -> f64 {
    let out: f64;
    unsafe {
        asm!("vmov.f64 d1, d0", out("d1") out, in("d0") x);
    }
    out
}
