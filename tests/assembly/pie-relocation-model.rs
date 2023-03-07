// revisions: x64
// assembly-output: emit-asm
// [x64] compile-flags: --target x86_64-unknown-linux-gnu -Crelocation-model=pie
// [x64] needs-llvm-components: x86


#![feature(no_core, lang_items)]
#![no_core]
#![crate_type="rlib"]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

// CHECK-LABEL: call_other_fn:
// With PIE local functions are called "directly".
// CHECK:       {{(jmp|callq)}} other_fn
#[no_mangle]
pub fn call_other_fn() -> u8 {
    unsafe {
        other_fn()
    }
}

// CHECK-LABEL: other_fn:
// External functions are still called through GOT, since we don't know if the symbol
// is defined in the binary or in the shared library.
// CHECK:       callq *foreign_fn@GOTPCREL(%rip)
#[no_mangle]
#[inline(never)]
pub fn other_fn() -> u8 {
    unsafe {
        foreign_fn()
    }
}

extern "C" {fn foreign_fn() -> u8;}
