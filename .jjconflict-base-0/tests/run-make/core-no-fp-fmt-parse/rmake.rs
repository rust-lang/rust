// This test checks that the core library of Rust can be compiled without enabling
// support for formatting and parsing floating-point numbers.

use run_make_support::{rustc, source_root};

fn main() {
    rustc()
        .edition("2024")
        .arg("-Dwarnings")
        .crate_type("rlib")
        .input(source_root().join("library/core/src/lib.rs"))
        .cfg("no_fp_fmt_parse")
        .run();
}
