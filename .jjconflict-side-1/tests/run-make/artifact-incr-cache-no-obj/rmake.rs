//@ needs-target-std
//
// emitting an object file is not necessary if user didn't ask for one
//
// This test is similar to run-make/artifact-incr-cache but it doesn't
// require to emit an object file
//
// Fixes: rust-lang/rust#123234

use run_make_support::rustc;

fn main() {
    for _ in 0..=1 {
        rustc()
            .input("lib.rs")
            .crate_type("lib")
            .emit("asm,dep-info,link,mir,llvm-ir,llvm-bc")
            .incremental("incremental")
            .run();
    }
}
