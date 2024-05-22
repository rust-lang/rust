// This test checks that the core library can still compile successfully
// when the no_global_oom_handling feature is turned on.
// See https://github.com/rust-lang/rust/pull/110649

use run_make_support::{rustc, tmp_dir};

fn main() {
    rustc()
        .edition("2021")
        .arg("-Dwarnings")
        .crate_type("rlib")
        .input("../../../library/core/src/lib.rs")
        .sysroot(tmp_dir().join("fakeroot"))
        .cfg("no_global_oom_handling")
        .run();
}
