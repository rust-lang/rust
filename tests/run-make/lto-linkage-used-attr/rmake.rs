// Link time optimizations (LTO) used to snip away some important symbols
// when setting optimization level to 3 or higher.
// This is an LLVM, not a rustc bug, fixed here: https://reviews.llvm.org/D145293
// This test checks that the impl_* symbols are preserved as they should.
// See https://github.com/rust-lang/rust/issues/108030

//@ only-x86_64-unknown-linux-gnu
// Reason: some of the inline assembly directives are architecture-specific.

use run_make_support::rustc;

fn main() {
    rustc().arg("-Cdebuginfo=0").opt_level("3").input("lib.rs").run();
    rustc().arg("-Clto=fat").arg("-Cdebuginfo=0").opt_level("3").input("main.rs").run();
}
