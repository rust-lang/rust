// Verifies that the `#[sanitize(address = "off")]` attribute also turns off
// the kernel address sanitizer.
//
//@ add-core-stubs
//@ compile-flags: -Zsanitizer=kernel-address -Ctarget-feature=-crt-static -Copt-level=0
//@ revisions: aarch64 riscv64imac riscv64gc x86_64
//@[aarch64] compile-flags: --target aarch64-unknown-none
//@[aarch64] needs-llvm-components: aarch64
//@[riscv64imac] compile-flags: --target riscv64imac-unknown-none-elf
//@[riscv64imac] needs-llvm-components: riscv
//@[riscv64gc] compile-flags: --target riscv64gc-unknown-none-elf
//@[riscv64gc] needs-llvm-components: riscv
//@[x86_64] compile-flags: --target x86_64-unknown-none
//@[x86_64] needs-llvm-components: x86

#![crate_type = "rlib"]
#![feature(no_core, sanitize, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: ; sanitize_off_asan_kasan::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[sanitize(address = "off")]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; sanitize_off_asan_kasan::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
