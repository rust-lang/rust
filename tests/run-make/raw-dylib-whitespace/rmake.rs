// Ensure that raw-dylib still works if the output directory contains spaces.

//@ only-windows-gnu
//@ ignore-cross-compile
//@ needs-dlltool
// Reason: this test specifically checks the dlltool feature,
// which is only used on windows-gnu.

use run_make_support::{rfs, rustc};

fn main() {
    let out_dir = std::path::absolute("path with spaces").unwrap();
    rfs::create_dir_all(&out_dir);
    rustc().crate_type("bin").input("main.rs").out_dir(&out_dir).env("TMP", &out_dir).run();
}
