// Checks that dynamically loading and unloading cdylib libraries works,
// both for simple functions and for functions that rely on the Rust runtime
// (that use thread local storage with destructors).
//
// - `foo.rs` is a cdylib.
// - `load_and_unload.rs` is a binary that loads and unloads the cdylib.
//
// This test is ignored on musl hosts because musl's dlclose is a no-op and
// therefore destructors won't run when the library is unloaded.
// See https://wiki.musl-libc.org/functional-differences-from-glibc.html#Unloading-libraries

//@ ignore-cross-compile
//@ ignore-musl

use run_make_support::{diff, run_with_args, rustc};

fn main() {
    rustc().input("foo.rs").run();

    rustc().crate_type("bin").crate_name("load_and_unload_bin").input("load_and_unload.rs").run();

    for command_arg in ["unload", "load_only"] {
        let out_raw = run_with_args("load_and_unload_bin", &[command_arg]).stdout_utf8();

        #[cfg(windows)]
        let output_filename = format!("output_{}_windows.txt", command_arg);
        #[cfg(unix)]
        let output_filename = format!("output_{}_unix.txt", command_arg);

        diff()
            .expected_file(output_filename)
            .actual_text("actual", out_raw)
            .normalize(r#"\r"#, "")
            .run();
    }
}
