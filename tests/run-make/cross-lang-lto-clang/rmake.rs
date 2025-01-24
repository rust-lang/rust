// This test checks that cross-language inlining actually works by checking
// the generated machine code.
// See https://github.com/rust-lang/rust/pull/57514

//@ needs-force-clang-based-tests
// NOTE(#126180): This test only runs on `x86_64-gnu-debug`, because that CI job sets
// RUSTBUILD_FORCE_CLANG_BASED_TESTS and only runs tests which contain "clang" in their
// name.

use run_make_support::{clang, env_var, llvm_ar, llvm_objdump, rustc, static_lib_name};

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
static RUST_ALWAYS_INLINED_PATTERN: &'static str = "bl.*<rust_always_inlined>";
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static RUST_ALWAYS_INLINED_PATTERN: &'static str = "call.*rust_always_inlined";
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
static C_ALWAYS_INLINED_PATTERN: &'static str = "bl.*<c_always_inlined>";
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static C_ALWAYS_INLINED_PATTERN: &'static str = "call.*c_always_inlined";

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
static RUST_NEVER_INLINED_PATTERN: &'static str = "bl.*<rust_never_inlined>";
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static RUST_NEVER_INLINED_PATTERN: &'static str = "call.*rust_never_inlined";
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
static C_NEVER_INLINED_PATTERN: &'static str = "bl.*<c_never_inlined>";
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static C_NEVER_INLINED_PATTERN: &'static str = "call.*c_never_inlined";

fn main() {
    rustc()
        .linker_plugin_lto("on")
        .output(static_lib_name("rustlib-xlto"))
        .opt_level("2")
        .codegen_units(1)
        .input("rustlib.rs")
        .run();
    clang()
        .lto("thin")
        .use_ld("lld")
        .arg("-lrustlib-xlto")
        .out_exe("cmain")
        .input("cmain.c")
        .arg("-O3")
        .run();
    // Make sure we don't find a call instruction to the function we expect to
    // always be inlined.
    llvm_objdump()
        .disassemble()
        .input("cmain")
        .run()
        .assert_stdout_not_contains_regex(RUST_ALWAYS_INLINED_PATTERN);
    // As a sanity check, make sure we do find a call instruction to a
    // non-inlined function
    llvm_objdump()
        .disassemble()
        .input("cmain")
        .run()
        .assert_stdout_contains_regex(RUST_NEVER_INLINED_PATTERN);
    clang().input("clib.c").lto("thin").arg("-c").out_exe("clib.o").arg("-O2").run();
    llvm_ar().obj_to_ar().output_input(static_lib_name("xyz"), "clib.o").run();
    rustc()
        .linker_plugin_lto("on")
        .opt_level("2")
        .linker(&env_var("CLANG"))
        .link_arg("-fuse-ld=lld")
        .input("main.rs")
        .output("rsmain")
        .run();
    llvm_objdump()
        .disassemble()
        .input("rsmain")
        .run()
        .assert_stdout_not_contains_regex(C_ALWAYS_INLINED_PATTERN);
    llvm_objdump()
        .disassemble()
        .input("rsmain")
        .run()
        .assert_stdout_contains_regex(C_NEVER_INLINED_PATTERN);
}
