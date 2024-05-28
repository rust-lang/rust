// After modifying the span of a function, if the length of
// the span remained the same but the end line number became different,
// this would cause an internal compiler error (ICE), fixed in #76256.

// This test compiles main.rs twice, first with end line 16 and
// then with end line 12. If compilation is successful, the end line
// was hashed by rustc in addition to the span length, and the fix still
// works.

// FIXME: Ignore flags temporarily disabled for the test.
// ignore-none
// ignore-nvptx64-nvidia-cuda

use run_make_support::{rustc, target, tmp_dir};
use std::fs;

fn main() {
    fs::create_dir(tmp_dir().join("src"));
    fs::create_dir(tmp_dir().join("incr"));
    fs::copy("a.rs", tmp_dir().join("main.rs"));
    rustc()
        .incremental(tmp_dir().join("incr"))
        .input(tmp_dir().join("src/main.rs"))
        .target(target())
        .run();
    fs::copy("b.rs", tmp_dir().join("main.rs"));
    rustc()
        .incremental(tmp_dir().join("incr"))
        .input(tmp_dir().join("src/main.rs"))
        .target(target())
        .run();
}
