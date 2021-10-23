// revisions: powerpc powerpc64 powerpc64le
//[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//[powerpc] needs-llvm-components: powerpc
//[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//[powerpc64] needs-llvm-components: powerpc
//[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//[powerpc64le] needs-llvm-components: powerpc

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

// CHECK-LABEL: @cr_clobber
// CHECK: call void asm sideeffect "", "~{cr}"()
#[no_mangle]
pub unsafe fn cr_clobber() {
    asm!("", out("cr") _, options(nostack, nomem));
}

// CHECK-LABEL: @cr0_clobber
// CHECK: call void asm sideeffect "", "~{cr0}"()
#[no_mangle]
pub unsafe fn cr0_clobber() {
    asm!("", out("cr0") _, options(nostack, nomem));
}

// CHECK-LABEL: @cr5_clobber
// CHECK: call void asm sideeffect "", "~{cr5}"()
#[no_mangle]
pub unsafe fn cr5_clobber() {
    asm!("", out("cr5") _, options(nostack, nomem));
}

// CHECK-LABEL: @xer_clobber
// CHECK: call void asm sideeffect "", "~{xer}"()
#[no_mangle]
pub unsafe fn xer_clobber() {
    asm!("", out("xer") _, options(nostack, nomem));
}
