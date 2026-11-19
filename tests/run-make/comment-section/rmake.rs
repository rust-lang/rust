// Both GCC and Clang write by default a `.comment` section with compiler information.
// Rustc received a similar .comment section, so this tests checks that this section
// properly appears.
// See https://github.com/rust-lang/rust/commit/74b8d324eb77a8f337b35dc68ac91b0c2c06debc

//@ only-linux
// FIXME(jieyouxu): check cross-compile setup
//@ ignore-cross-compile
// FIXME: remove the ignore gcc once fixed.
//@ ignore-backends: gcc

use run_make_support::{cwd, env_var, llvm_readobj, rfs, rustc};

fn main() {
    let target = env_var("TARGET");

    rustc()
        .arg("-")
        .stdin_buf("fn main() {}")
        .emit("link,obj")
        .arg("-Csave-temps")
        .target(target)
        .run();

    // Check linked output has a `.comment` section with the expected content.
    llvm_readobj()
        .section(".comment")
        .input("rust_out")
        .run()
        .assert_stdout_contains("rustc version 1.");

    // Check codegen object files (including temporary outputs) have a `.comment`
    // section with the expected content.
    rfs::read_dir_entries(cwd(), |f| {
        // Synthetic linker inputs are not codegen objects and have no `.comment`.
        let name = f.file_name().unwrap().to_string_lossy();
        if !f.extension().is_some_and(|ext| ext == "o")
            || name.contains(".rustc_link_metadata.")
            || name.contains(".rustc_link_symbols.")
        {
            return;
        }

        llvm_readobj()
            .section(".comment")
            .input(&f)
            .run()
            .assert_stdout_contains("rustc version 1.");
    });
}
