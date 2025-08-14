//@ needs-target-std
//
// rustc should be able to emit required files (asm, llvm-*, etc) during incremental
// compilation on the first pass by running the code gen as well as on subsequent runs -
// extracting them from the cache
//
// Fixes: rust-lang/rust#89149
// Fixes: rust-lang/rust#88829
// Also see discussion at
// <https://internals.rust-lang.org/t/interaction-between-incremental-compilation-and-emit/20551>

use run_make_support::rustc;

fn main() {
    for _ in 0..=1 {
        rustc()
            .input("lib.rs")
            .crate_type("lib")
            .emit("obj,asm,dep-info,link,mir,llvm-ir,llvm-bc")
            .incremental("incremental")
            .run();
    }
}
