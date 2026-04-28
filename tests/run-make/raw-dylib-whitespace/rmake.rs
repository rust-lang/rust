// Ensure that raw-dylib still works if the output directory contains spaces.

//@ only-windows-gnu
//@ ignore-cross-compile

use run_make_support::{rfs, rustc};

fn main() {
    let out_dir = "path with spaces";
    rfs::create_dir_all(out_dir);
    unsafe {
        std::env::set_var("TMP", out_dir);
    }
    rustc().crate_type("bin").input("main.rs").out_dir(out_dir).run();
}
