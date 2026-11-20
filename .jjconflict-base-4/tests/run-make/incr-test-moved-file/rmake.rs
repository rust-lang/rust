// The generated test harness code contains spans with a dummy location,
// but a non-dummy SyntaxContext. Previously, the incremental cache was encoding
// these spans as a full span (with a source file index), instead of skipping
// the encoding of the location information. If the file gest moved, the hash
// of the span will be unchanged (since it has a dummy location), so the incr
// cache would end up try to load a non-existent file using the previously
// enccoded source file id.
// This test reproduces the steps that used to trigger this bug, and checks
// for successful compilation.
// See https://github.com/rust-lang/rust/issues/83112

//@ ignore-cross-compile

use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_dir("incr");
    rfs::create_dir("src");
    rfs::create_dir("src/mydir");
    rfs::copy("main.rs", "src/main.rs");
    rustc().input("src/main.rs").incremental("incr").arg("--test").run();
    rfs::rename("src/main.rs", "src/mydir/main.rs");
    rustc().input("src/mydir/main.rs").incremental("incr").arg("--test").run();
}
