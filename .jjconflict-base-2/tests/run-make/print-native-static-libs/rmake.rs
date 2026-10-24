//! This checks the output of `--print=native-static-libs`
//!
//! Specifically, this test makes sure that one and only one
//! note is emitted with the text "native-static-libs:" as prefix
//! that the note contains the link args given in the source code
//! and cli of the current crate and downstream crates.
//!
//! It also checks that there aren't any duplicated consecutive
//! args, as they are useless and suboptimal for debugability.
//! See https://github.com/rust-lang/rust/issues/113209.

//@ ignore-cross-compile
//@ ignore-wasm

use run_make_support::{is_windows_msvc, rustc};

fn main() {
    // build supporting crate
    rustc().input("bar.rs").crate_type("rlib").arg("-lbar_cli").run();

    // build main crate as staticlib
    let output = rustc()
        .input("foo.rs")
        .crate_type("staticlib")
        .arg("-lfoo_cli")
        .arg("-lfoo_cli") // 2nd time
        .print("native-static-libs")
        .run();

    let mut found_note = false;
    for l in output.stderr_utf8().lines() {
        let Some(args) = l.strip_prefix("note: native-static-libs:") else {
            continue;
        };
        assert!(!found_note);
        found_note = true;

        let args: Vec<&str> = args.trim().split_ascii_whitespace().collect();

        macro_rules! assert_contains_lib {
            ($lib:literal in $args:ident) => {{
                let lib = format!(
                    "{}{}{}",
                    if !is_windows_msvc() { "-l" } else { "" },
                    $lib,
                    if !is_windows_msvc() { "" } else { ".lib" },
                );
                let found = $args.contains(&&*lib);
                assert!(found, "unable to find lib `{}` in those linker args: {:?}", lib, $args);
            }};
        }

        assert_contains_lib!("glib-2.0" in args); // in bar.rs
        assert_contains_lib!("systemd" in args); // in foo.rs
        assert_contains_lib!("bar_cli" in args);
        assert_contains_lib!("foo_cli" in args);

        // make sure that no args are consecutively present
        let dedup_args: Vec<&str> = {
            let mut args = args.clone();
            args.dedup();
            args
        };
        assert_eq!(args, dedup_args);
    }

    assert!(found_note);
}
