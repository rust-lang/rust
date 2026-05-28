// This test ensures that if LTO occurs between crates with different DWARF versions, we
// will choose the highest DWARF version for the final binary. This matches Clang's behavior.
// Note: `.2byte` directive is used on MIPS.
// Note: `.half` directive is used on RISC-V.

//@ only-linux
//@ aux-build:dwarf-mixed-versions-lto-aux.rs
//@ compile-flags: -C lto -g -Cdwarf-version=5
//@ assembly-output: emit-asm
//@ no-prefer-dynamic

extern crate dwarf_mixed_versions_lto_aux;

fn main() {
    dwarf_mixed_versions_lto_aux::check_is_even(&0);
}

// CHECK: .section .debug_info
// CHECK-NOT: {{\.(short|hword|2byte|half)}} 2
// CHECK-NOT: {{\.(short|hword|2byte|half)}} 4
// CHECK: {{\.(short|hword|2byte|half)}} 5
