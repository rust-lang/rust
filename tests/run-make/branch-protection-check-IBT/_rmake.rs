// Check for GNU Property Note

// How to run this
// python3 x.py test --target x86_64-unknown-linux-gnu  tests/run-make/branch-protection-check-IBT/

//@ only-x86_64

//@ ignore-test
// FIXME(jieyouxu): see the FIXME in the Makefile

use run_make_support::llvm_readobj;
use run_make_support::rustc;
use run_make_support::{cwd, env_var};

fn main() {
    let llvm_components = env_var("LLVM_COMPONENTS");
    if !format!(" {llvm_components} ").contains(" x86 ") {
        return;
    }

    rustc()
        .input("main.rs")
        .target("x86_64-unknown-linux-gnu")
        .arg("-Zcf-protection=branch")
        .arg(format!("-L{}", cwd().display()))
        .arg("-Clink-args=-nostartfiles")
        .arg("-Csave-temps")
        .run();

    llvm_readobj().arg("-nW").input("main").run().assert_stdout_contains(".note.gnu.property");
}
