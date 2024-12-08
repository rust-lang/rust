//@ revisions: hexagon
//@[hexagon] compile-flags: --target hexagon-unknown-linux-musl
//@[hexagon] needs-llvm-components: hexagon
//@ compile-flags: -Zmerge-functions=disabled

#![crate_type = "rlib"]
#![feature(no_core, rustc_attrs, lang_items, asm_experimental_arch)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

// CHECK-LABEL: @flags_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn flags_clobber() {
    asm!("", options(nostack, nomem));
}

// CHECK-LABEL: @no_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn no_clobber() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @p0_clobber
// CHECK: call void asm sideeffect "", "~{p0}"()
#[no_mangle]
pub unsafe fn p0_clobber() {
    asm!("", out("p0") _, options(nostack, nomem, preserves_flags));
}
