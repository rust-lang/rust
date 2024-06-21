// Both GCC and Clang write by default a `.comment` section with compiler information.
// Rustc received a similar .comment section, so this tests checks that this section
// properly appears.
// See https://github.com/rust-lang/rust/commit/74b8d324eb77a8f337b35dc68ac91b0c2c06debc

//@ only-linux
// FIXME(jieyouxu): check cross-compile setup
//@ ignore-cross-compile

use std::path::PathBuf;

use run_make_support::llvm_readobj;
use run_make_support::rustc;
use run_make_support::{cwd, env_var, read_dir, run_in_tmpdir};

fn main() {
    let target = env_var("TARGET");

    rustc()
        .arg("-")
        .stdin("fn main() {}")
        .emit("link,obj")
        .arg("-Csave-temps")
        .target(&target)
        .run();

    // Check linked output has a `.comment` section with the expected content.
    llvm_readobj()
        .section(".comment")
        .input("rust_out")
        .run()
        .assert_stdout_contains("rustc version 1.");

    // Check all object files (including temporary outputs) have a `.comment`
    // section with the expected content.
    read_dir(cwd(), |f| {
        if !f.extension().is_some_and(|ext| ext == "o") {
            return;
        }

        llvm_readobj()
            .section(".comment")
            .input(&f)
            .run()
            .assert_stdout_contains("rustc version 1.");
    });
}
