// Checks that dynamically loading and unloading cdylib libraries works,
// both for simple functions and for functions that rely on the Rust runtime
// (that use thread local storage with destructors).
//
// - `foo.rs` is a cdylib.
// - `load_and_unload.rs` is a binary that loads and unloads the cdylib.
//

//@ ignore-cross-compile

use run_make_support::{diff, run_with_args, rustc};

// On musl and AIX, `dlclose` does not invoke TLS destructors upon unloading,
// so only the `load_only` case is tested on those platforms.
//
// Reference for musl:
// https://wiki.musl-libc.org/functional-differences-from-glibc.html#Unloading-libraries

#[cfg(any(target_env = "musl", target_os = "aix"))]
const TEST_MODES: &[&str] = &["load_only"];

#[cfg(not(any(target_env = "musl", target_os = "aix")))]
const TEST_MODES: &[&str] = &["unload", "load_only"];

fn main() {
    rustc().input("foo.rs").run();

    rustc().crate_type("bin").crate_name("load_and_unload_bin").input("load_and_unload.rs").run();

    for command_arg in TEST_MODES {
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
