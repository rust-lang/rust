// Verifies that `-Zsanitizer=kernel-address` emits sanitizer instrumentation.

//@ add-core-stubs
//@ compile-flags: -Zsanitizer=kernel-address -Copt-level=0
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

// CHECK-LABEL: ; kasan_emits_instrumentation::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[sanitize(address = "off")]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; kasan_emits_instrumentation::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
