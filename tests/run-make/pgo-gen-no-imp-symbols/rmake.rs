//@ needs-target-std
//
// LLVM's profiling instrumentation adds a few symbols that are used by the profiler runtime.
// Since these show up as globals in the LLVM IR, the compiler generates dllimport-related
// __imp_ stubs for them. This can lead to linker errors because the instrumentation
// symbols have weak linkage or are in a comdat section, but the __imp_ stubs aren't.
// Since profiler-related symbols were excluded from stub-generation in #59812, this has
// been fixed, and this test checks that the llvm profile symbol appear, but without the
// anomalous __imp_ stubs.
// See https://github.com/rust-lang/rust/pull/59812

use run_make_support::{cwd, rfs, rustc};

fn main() {
    rustc()
        .input("test.rs")
        .emit("llvm-ir")
        .opt()
        .codegen_units(1)
        .profile_generate(cwd())
        .arg("-Zno-profiler-runtime")
        .run();
    let out = rfs::read_to_string("test.ll");
    // We expect symbols starting with "__llvm_profile_".
    assert!(out.contains("__llvm_profile_"));
    // We do NOT expect the "__imp_" version of these symbols.
    assert!(!out.contains("__imp___llvm_profile_")); // 64 bit
    assert!(!out.contains("__imp____llvm_profile_")); // 32 bit
}
