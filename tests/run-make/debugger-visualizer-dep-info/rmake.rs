//@ needs-target-std
//
// This test checks that files referenced via #[debugger_visualizer] are
// included in `--emit dep-info` output.
// See https://github.com/rust-lang/rust/pull/111641

use run_make_support::{invalid_utf8_contains, rustc};

fn main() {
    rustc().emit("dep-info").input("main.rs").run();
    invalid_utf8_contains("main.d", "my_gdb_script.py");
    invalid_utf8_contains("main.d", "my_visualizers/bar.natvis");
}
