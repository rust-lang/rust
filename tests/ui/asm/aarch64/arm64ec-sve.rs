//@ compile-flags: --target arm64ec-pc-windows-msvc
//@ needs-asm-support
//@ needs-llvm-components: aarch64

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items, asm_experimental_arch)]
#![no_core]

// SVE cannot be used for Arm64EC
// https://github.com/rust-lang/rust/pull/131332#issuecomment-2401189142

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
        asm!("", out("z0") _, in("z1") x);
        //~^ ERROR invalid register `z0`: SVE cannot be used for Arm64EC
        //~^^ ERROR invalid register `z1`: SVE cannot be used for Arm64EC
        asm!("", out("p0") _);
        //~^ ERROR invalid register `p0`: SVE cannot be used for Arm64EC
        asm!("", out("ffr") _);
        //~^ ERROR invalid register `ffr`: SVE cannot be used for Arm64EC
    }
}
