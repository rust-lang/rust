// This test checks that the core library can still compile successfully
// when the no_global_oom_handling feature is turned on.
// See https://github.com/rust-lang/rust/pull/110649

use run_make_support::{rustc, source_root};

fn main() {
    rustc()
        .edition("2024")
        .arg("-Dwarnings")
        .crate_type("rlib")
        .input(source_root().join("library/core/src/lib.rs"))
        .sysroot("fakeroot")
        .cfg("no_global_oom_handling")
        .run();
}
