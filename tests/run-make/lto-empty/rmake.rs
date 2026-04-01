// Compiling Rust code twice in a row with "fat" link-time-optimizations used to cause
// an internal compiler error (ICE). This was due to how the compiler would cache some modules
// to make subsequent compilations faster, at least one of which was required for LTO to link
// into. After this was patched in #63956, this test checks that the bug does not make
// a resurgence.
// See https://github.com/rust-lang/rust/issues/63349

//@ ignore-cross-compile

use run_make_support::rustc;

fn main() {
    rustc().input("lib.rs").arg("-Clto=fat").opt_level("3").incremental("inc-fat").run();
    rustc().input("lib.rs").arg("-Clto=fat").opt_level("3").incremental("inc-fat").run();
    rustc().input("lib.rs").arg("-Clto=thin").opt_level("3").incremental("inc-thin").run();
    rustc().input("lib.rs").arg("-Clto=thin").opt_level("3").incremental("inc-thin").run();
}
