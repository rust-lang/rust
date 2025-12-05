// `-Z branch protection` is an unstable compiler feature which adds pointer-authentication
// code (PAC), a useful hashing measure for verifying that pointers have not been modified.
// This test checks that compilation and execution is successful when this feature is activated,
// with some of its possible extra arguments (bti, gcs, pac-ret, pc, leaf, b-key).
// See https://github.com/rust-lang/rust/pull/88354

//@ only-aarch64
// Reason: branch protection is not supported on other architectures
//@ ignore-apple
// Reason: XCode needs updating to support gcs
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, cc, is_windows_msvc, llvm_ar, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().arg("-Zbranch-protection=bti,gcs,pac-ret,leaf").input("test.rs").run();
    run("test");
    cc().arg("-v")
        .arg("-c")
        .out_exe("test")
        .input("test.c")
        .arg("-mbranch-protection=bti+gcs+pac-ret+leaf")
        .run();
    let obj_file = if is_windows_msvc() { "test.obj" } else { "test" };
    llvm_ar().obj_to_ar().output_input("libtest.a", &obj_file).run();
    rustc().arg("-Zbranch-protection=bti,gcs,pac-ret,leaf").input("test.rs").run();
    run("test");

    // FIXME: +pc was only recently added to LLVM
    // cc().arg("-v")
    //     .arg("-c")
    //     .out_exe("test")
    //     .input("test.c")
    //     .arg("-mbranch-protection=bti+pac-ret+pc+leaf")
    //     .run();
    // let obj_file = if is_windows_msvc() { "test.obj" } else { "test" };
    // llvm_ar().obj_to_ar().output_input("libtest.a", &obj_file).run();
    // rustc().arg("-Zbranch-protection=bti,pac-ret,pc,leaf").input("test.rs").run();
    // run("test");
}
