// Verifies that HWASAN and KHWASAN emit different instrumentation.
//
//@ add-minicore
//@ revisions: hwasan khwasan
//@[hwasan] compile-flags: -Zsanitizer=hwaddress
//@[khwasan] compile-flags: -Zsanitizer=kernel-hwaddress
//@ compile-flags: --target aarch64-unknown-none -Copt-level=0
//@ needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(no_core, lang_items, sanitize)]
#![no_core]

extern crate minicore;

// hwasan-LABEL: define {{.*}} @test
// hwasan:       @__hwasan_tls
// hwasan:       call void @llvm.hwasan.check.memaccess.shortgranules
// hwasan:       declare void @__hwasan_init()

// khwasan-LABEL: define {{.*}} @test
// khwasan-NOT:   @__hwasan_init
// khwasan:       @__hwasan_tls
// khwasan:       call void @llvm.hwasan.check.memaccess.shortgranules
#[no_mangle]
pub fn test(b: &mut u8) -> u8 {
    *b
}
