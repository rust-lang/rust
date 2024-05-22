// This test checks that alloc can still compile successfully
// when the unstable no_sync feature is turned on.
// See https://github.com/rust-lang/rust/pull/84266

use run_make_support::rustc;

fn main() {
    rustc()
        .edition("2021")
        .arg("-Dwarnings")
        .crate_type("rlib")
        .input("../../../library/alloc/src/lib.rs")
        .cfg("no_sync")
        .run();
}
