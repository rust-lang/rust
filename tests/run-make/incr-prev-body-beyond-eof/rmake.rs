// After modifying the span of a function, if the length of
// the span remained the same but the end line number became different,
// this would cause an internal compiler error (ICE), fixed in #76256.

// This test compiles main.rs twice, first with end line 16 and
// then with end line 12. If compilation is successful, the end line
// was hashed by rustc in addition to the span length, and the fix still
// works.

//@ ignore-cross-compile

use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_dir("src");
    rfs::create_dir("incr");
    rfs::copy("a.rs", "src/main.rs");
    rustc().incremental("incr").input("src/main.rs").run();
    rfs::copy("b.rs", "src/main.rs");
    rustc().incremental("incr").input("src/main.rs").run();
}
