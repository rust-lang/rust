#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate dep;

unsafe extern "C" {
    pub safe static VAR: i32;
}

pub mod mod_0 {
    use super::VAR;
    #[no_mangle]
    pub fn refer_0() -> i32 {
        VAR
    }
}

pub mod mod_1 {
    use super::VAR;
    #[no_mangle]
    pub fn refer_1() -> i32 {
        VAR
    }
}

#[no_mangle]
pub fn call_dep() -> i32 {
    dep::refer_dep()
}

// DEFAULT: @VAR = {{.*}}dso_local{{.*}}global i32
// DEFAULT-NOT: direct-access-external-data
// DEFAULT-NOT: PIE Level

// PIE: @VAR = external
// PIE-NOT: dso_local
// PIE-SAME: global i32
// PIE-NOT: direct-access-external-data
// PIE: !{{[0-9]+}} = !{i32 7, !"PIE Level", i32 2}

// DIRECT: @VAR = {{.*}}dso_local{{.*}}global i32
// DIRECT-DAG: !{{[0-9]+}} = !{i32 7, !"direct-access-external-data", i32 1}
// DIRECT-DAG: !{{[0-9]+}} = !{i32 7, !"PIE Level", i32 2}

// INDIRECT: @VAR = external
// INDIRECT-NOT: dso_local
// INDIRECT-SAME: global i32
// INDIRECT: !{{[0-9]+}} = !{i32 7, !"direct-access-external-data", i32 0}
// INDIRECT-NOT: PIE Level

// FIXME: Under normal circumstances PIE Level should be set consistently, so
// GOT relocations don't get emitted with direct-access-external-data enabled
// for all versions of Full/Thin LTO.
// DEP-NO-PIE-NOT: PIE Level

// Demonstrate incorrect GOT relocation under direct-access-external-data for ThinLTO.
// Even with direct-access enabled, the missing PIE Level on the dependency's module
// causes LLVM to fall back to GOT indirection.
// DIRECT-RELOC-THIN: R_X86_64_GOTPCREL{{.*}}VAR

// For Full LTO we expect direct PC-relative access since the merged module
// correctly inherits the main module's PIE Level.
// DIRECT-RELOC-FAT-NOT: R_X86_64_GOTPCREL{{.*}}VAR
// DIRECT-RELOC-FAT: R_X86_64_PC32{{.*}}VAR

// For indirect cases, we always expect GOT indirection.
// INDIRECT-RELOC: R_X86_64_GOTPCREL{{.*}}VAR
// INDIRECT-RELOC-NOT: R_X86_64_PC32{{.*}}VAR

// Default PIE without direct-access enabled should use GOT indirection.
// This is correct and is not changed by setting PIE Level consistently.
// PIE-RELOC: R_X86_64_GOTPCREL{{.*}}VAR
// PIE-RELOC-NOT: R_X86_64_PC32{{.*}}VAR
