// The rust crate foo will link to the native library foo, while the rust crate
// bar will link to the native library bar. There is also a dependency between
// the native library bar to the natibe library foo.
// This test ensures that the ordering of -lfoo and -lbar on the command line is
// correct to complete the linkage. If passed as "-lfoo -lbar", then the 'foo'
// library will be stripped out, and the linkage will fail.
// See https://github.com/rust-lang/rust/commit/e6072fa0c4c22d62acf3dcb78c8ee260a1368bd7

//@ ignore-cross-compile
// Reason: linkage still fails as the object files produced are not in the correct
// format in the `build_native_static_lib` step

use run_make_support::{build_native_static_lib, rustc};

fn main() {
    build_native_static_lib("foo");
    build_native_static_lib("bar");
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").run();
    rustc().input("main.rs").print("link-args").run();
}
