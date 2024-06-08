// After modifying the span of a function, if the length of
// the span remained the same but the end line number became different,
// this would cause an internal compiler error (ICE), fixed in #76256.

// This test compiles main.rs twice, first with end line 16 and
// then with end line 12. If compilation is successful, the end line
// was hashed by rustc in addition to the span length, and the fix still
// works.

//@ ignore-none
// reason: no-std is not supported

//@ ignore-nvptx64-nvidia-cuda
// FIXME: can't find crate for `std`

use run_make_support::rustc;
use std::fs;

fn main() {
    // FIXME(Oneirical): Use run_make_support::fs_wrapper here.
    fs::create_dir("src").unwrap();
    fs::create_dir("incr").unwrap();
    fs::copy("a.rs", "src/main.rs").unwrap();
    rustc().incremental("incr").input("src/main.rs").run();
    fs::copy("b.rs", "src/main.rs").unwrap();
    rustc().incremental("incr").input("src/main.rs").run();
}
