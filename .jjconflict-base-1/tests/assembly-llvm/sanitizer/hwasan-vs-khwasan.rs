// Verifies that HWASAN and KHWASAN emit different assembly instrumentation on AArch64.
//
//@ add-minicore
//@ assembly-output: emit-asm
//@ revisions: hwasan khwasan
//@[hwasan] compile-flags: --target aarch64-unknown-linux-gnu -Zsanitizer=hwaddress
//@[hwasan] needs-llvm-components: aarch64
//@[khwasan] compile-flags: --target aarch64-unknown-none -Zsanitizer=kernel-hwaddress
//@[khwasan] needs-llvm-components: aarch64
//@ compile-flags: -Copt-level=1

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;

// hwasan-LABEL: test:
// hwasan:       adrp    x{{[0-9]+}}, :gottprel:__hwasan_tls
// hwasan:       mrs     x{{[0-9]+}}, TPIDR_EL0
// hwasan:       bl      __hwasan_check_x0_0_short_v2

// khwasan-LABEL: test:
// khwasan-NOT:   __hwasan_tls
// khwasan:       orr     x{{[0-9]+}}, x0, #0xff00000000000000
// khwasan:       bl      __hwasan_check_x0_67043328_fixed_0_short_v2

#[no_mangle]
pub fn test(b: &mut u8) -> u8 {
    *b
}
