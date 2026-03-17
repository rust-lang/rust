// Verifies that HWASAN and KHWASAN emit different instrumentation.
//
//@ add-minicore
//@ revisions: hwasan khwasan
//@[hwasan] compile-flags: --target aarch64-unknown-linux-gnu -Zsanitizer=hwaddress
//@[hwasan] needs-llvm-components: aarch64
//@[khwasan] compile-flags: --target aarch64-unknown-none -Zsanitizer=kernel-hwaddress
//@[khwasan] needs-llvm-components: aarch64
//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]
#![feature(no_core, lang_items, sanitize)]
#![no_core]

extern crate minicore;

// hwasan-LABEL: define {{.*}} @test
// hwasan:       @__hwasan_tls
// hwasan:       call void @llvm.hwasan.check.memaccess.shortgranules
// hwasan:       declare void @__hwasan_init()

// The `__hwasan_tls` symbol is unconditionally declared by LLVM's `HWAddressSanitizer` pass.
// However, in kernel mode KHWASAN does not actually use it (because shadow mapping is fixed
// and the stack frame ring buffer is disabled). It remains an unused declaration in the LLVM IR
// and is optimized out before the final assembly/object file is generated, so it does not end
// up in the final binary. Thus, assert that it appears in the output, but not inside `test`.
//
// khwasan:       @__hwasan_tls
// khwasan-LABEL: define {{.*}} @test
// khwasan-NOT:   @__hwasan_tls
//
// Also test a few other things appear under the LABEL.
//
// khwasan-NOT:   @__hwasan_init
// khwasan:       call void @llvm.hwasan.check.memaccess.shortgranules
#[no_mangle]
pub fn test(b: &mut u8) -> u8 {
    *b
}
