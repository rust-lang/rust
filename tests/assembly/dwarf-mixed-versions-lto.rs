// This test ensures that if LTO occurs between crates with different DWARF versions, we
// will choose the highest DWARF version for the final binary. This matches Clang's behavior.

//@ only-linux
//@ aux-build:dwarf-mixed-versions-lto-aux.rs
//@ compile-flags: -C lto -g -Zdwarf-version=5
//@ assembly-output: emit-asm
//@ no-prefer-dynamic

extern crate dwarf_mixed_versions_lto_aux;

fn main() {
    dwarf_mixed_versions_lto_aux::check_is_even(&0);
}

// CHECK: .section .debug_info
// CHECK-NOT: {{\.(short|hword)}} 2
// CHECK-NOT: {{\.(short|hword)}} 4
// CHECK: {{\.(short|hword)}} 5
