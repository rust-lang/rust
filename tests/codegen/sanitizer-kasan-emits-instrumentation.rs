// Verifies that `-Zsanitizer=kernel-address` emits sanitizer instrumentation.

// compile-flags: -Zsanitizer=kernel-address
// revisions: aarch64 riscv64imac riscv64gc x86_64
//[aarch64] compile-flags: --target aarch64-unknown-none
//[aarch64] needs-llvm-components: aarch64
//[riscv64imac] compile-flags: --target riscv64imac-unknown-none-elf
//[riscv64imac] needs-llvm-components: riscv
//[riscv64imac] min-llvm-version: 16
//[riscv64gc] compile-flags: --target riscv64gc-unknown-none-elf
//[riscv64gc] needs-llvm-components: riscv
//[riscv64gc] min-llvm-version: 16
//[x86_64] compile-flags: --target x86_64-unknown-none
//[x86_64] needs-llvm-components: x86

#![crate_type = "rlib"]
#![feature(no_core, no_sanitize, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

impl Copy for u8 {}

// CHECK-LABEL: ; sanitizer_kasan_emits_instrumentation::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[no_sanitize(address)]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; sanitizer_kasan_emits_instrumentation::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}
