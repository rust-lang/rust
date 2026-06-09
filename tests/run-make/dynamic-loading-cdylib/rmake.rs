// Checks that dynamically loading and unloading cdylib libraries works,
// both for simple functions and for functions that rely on the Rust runtime
// (that use thread local storage with destructors).
//
// - `foo.rs` is a cdylib.
// - `load_and_unload.rs` is a binary that loads and unloads the cdylib.

//@ ignore-cross-compile

use run_make_support::{diff, run, rustc};

fn main() {
    rustc().input("foo.rs").run();

    rustc().crate_type("bin").crate_name("load_and_unload_bin").input("load_and_unload.rs").run();

    let out_raw = run("load_and_unload_bin").stdout_utf8();

    #[cfg(windows)]
    let output_filename = "output_windows.txt";
    #[cfg(unix)]
    let output_filename = "output_unix.txt";

    diff()
        .expected_file(output_filename)
        .actual_text("actual", out_raw)
        .normalize(r#"\r"#, "")
        .run();
}
