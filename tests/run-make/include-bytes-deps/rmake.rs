//@ needs-target-std
//
// include_bytes! and include_str! in `main.rs`
// should register the included file as of #24423,
// and this test checks that this is still the case.
// See https://github.com/rust-lang/rust/pull/24423

use run_make_support::{invalid_utf8_contains, rustc};

fn main() {
    rustc().emit("dep-info").input("main.rs").run();
    invalid_utf8_contains("main.d", "input.txt");
    invalid_utf8_contains("main.d", "input.bin");
    invalid_utf8_contains("main.d", "input.md");
}
