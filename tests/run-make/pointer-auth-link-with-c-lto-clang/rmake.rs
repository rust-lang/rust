// `-Z branch protection` is an unstable compiler feature which adds pointer-authentication
// code (PAC), a useful hashing measure for verifying that pointers have not been modified.
// This test checks that compilation and execution is successful when this feature is activated,
// with some of its possible extra arguments (bti, gcs, pac-ret, leaf) when doing LTO.
// See https://github.com/rust-lang/rust/pull/88354

//@ needs-force-clang-based-tests
//@ only-aarch64
// Reason: branch protection is not supported on other architectures
//@ ignore-apple
// Reason: XCode needs updating to support gcs
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{clang, env_var, llvm_ar, llvm_objdump, run, rustc, static_lib_name};

static PAUTH_A_KEY_PATTERN: &'static str = "paciasp";
static PAUTH_B_KEY_PATTERN: &'static str = "pacibsp";

fn main() {
    clang()
        .arg("-v")
        .lto("thin")
        .arg("-mbranch-protection=bti+gcs+pac-ret+b-key+leaf")
        .arg("-c")
        .out_exe("test.o")
        .input("test.c")
        .run();
    llvm_ar().obj_to_ar().output_input(static_lib_name("test"), "test.o").run();
    rustc()
        .linker_plugin_lto("on")
        .opt_level("2")
        .linker(&env_var("CLANG"))
        .link_arg("-fuse-ld=lld")
        .arg("-Zbranch-protection=bti,gcs,pac-ret,leaf")
        .input("test.rs")
        .output("test.bin")
        .run();

    // Check that both a-key and b-key pac-ret survived LTO
    llvm_objdump()
        .disassemble()
        .input("test.bin")
        .run()
        .assert_stdout_contains_regex(PAUTH_A_KEY_PATTERN)
        .assert_stdout_contains_regex(PAUTH_B_KEY_PATTERN);

    // Check that the binary actually runs
    run("test.bin");
}
