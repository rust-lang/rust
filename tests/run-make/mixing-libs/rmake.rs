// Having multiple upstream crates available in different formats
// should result in failed compilation. This test causes multiple
// libraries to exist simultaneously as rust libs and dynamic libs,
// causing prog.rs to fail compilation.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile

use run_make_support::{dynamic_lib_name, rfs, rustc};

fn main() {
    rustc().input("rlib.rs").crate_type("rlib").crate_type("dylib").run();

    // Not putting `-C prefer-dynamic` here allows for static linking of librlib.rlib.
    rustc().input("dylib.rs").run();

    // librlib's dynamic version needs to be removed here to prevent prog.rs from fetching
    // the wrong one.
    rfs::remove_file(dynamic_lib_name("rlib"));
    rustc().input("prog.rs").run_fail();
}
