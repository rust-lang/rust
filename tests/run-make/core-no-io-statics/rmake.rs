// This test checks that the core library of Rust can be compiled when enabling
// `--cfg no_io_statics`.

use run_make_support::{rustc, source_root};

fn main() {
    rustc()
        .edition("2024")
        .arg("-Dwarnings")
        .crate_type("rlib")
        .input(source_root().join("library/core/src/lib.rs"))
        .cfg("no_io_statics")
        .run();
}
