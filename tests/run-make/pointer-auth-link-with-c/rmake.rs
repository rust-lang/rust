// `-Z branch protection` is an unstable compiler feature which adds pointer-authentication
// code (PAC), a useful hashing measure for verifying that pointers have not been modified.
// This test checks that compilation and execution is successful when this feature is activated,
// with some of its possible extra arguments (bti, pac-ret, leaf).
// See https://github.com/rust-lang/rust/pull/88354

//@ only-aarch64
// Reason: branch protection is not supported on other architectures
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, cc, is_msvc, llvm_ar, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().arg("-Zbranch-protection=bti,pac-ret,leaf").input("test.rs").run();
    run("test");
    cc().arg("-v")
        .arg("-c")
        .out_exe("test")
        .input("test.c")
        .arg("-mbranch-protection=bti+pac-ret+leaf")
        .run();
    let obj_file = if is_msvc() { "test.obj" } else { "test" };
    llvm_ar().obj_to_ar().output_input("libtest.a", &obj_file).run();
    rustc().arg("-Zbranch-protection=bti,pac-ret,leaf").input("test.rs").run();
    run("test");
}
