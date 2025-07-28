//@ only-x86_64
//@ revisions: base avx512
//@ [avx512]compile-flags: -C target-feature=+avx512f

#![crate_type = "rlib"]

use std::arch::asm;

// CHECK-LABEL: @amx_clobber
// base: call void asm sideeffect inteldialect "", "~{tmm0}"()
#[no_mangle]
pub unsafe fn amx_clobber() {
    asm!("", out("tmm0") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @avx512_clobber
// base: call void asm sideeffect inteldialect "", "~{xmm31}"()
// avx512: call float asm sideeffect inteldialect "", "=&{xmm31}"()
#[no_mangle]
pub unsafe fn avx512_clobber() {
    asm!("", out("zmm31") _, options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @eax_clobber
// CHECK: call i32 asm sideeffect inteldialect "", "=&{ax}"()
#[no_mangle]
pub unsafe fn eax_clobber() {
    asm!("", out("eax") _, options(nostack, nomem, preserves_flags));
}
