//@ ignore-cross-compile
//@ needs-crate-type: proc-macro

// Regression test for <https://github.com/rust-lang/rust/issues/95945>.
// Recompiling incrementally after inserting an enum variant before an existing one used
// to ICE with "Found unstable fingerprints for def_ident_span". The derive re-emits the
// variant identifiers as associated constants, so their spans move while the surrounding
// generated tokens keep call-site hygiene.
//
// This cannot use the `revisions` system: the `#[cfg]`-based revisions keep both versions
// of the text in the file, so the identifier spans never move. The source has to actually
// be rewritten between the two compilations.

use std::fs;
use std::path::PathBuf;

use run_make_support::{rfs, rustc};

fn main() {
    rustc().input("macros/lib.rs").crate_name("macros").crate_type("proc-macro").run();
    let macros_dylib = find_proc_macro_dylib("macros");

    rfs::write("lib.rs", "#[derive(macros::Bar)]\npub enum FooEnum { One }\n");
    rustc()
        .input("lib.rs")
        .crate_type("lib")
        .incremental("incr")
        .arg("-Zincremental-verify-ich")
        .extern_("macros", &macros_dylib)
        .run();

    // Insert a variant *before* the existing one, moving `One`'s span.
    rfs::write("lib.rs", "#[derive(macros::Bar)]\npub enum FooEnum { Zero, One }\n");
    let out = rustc()
        .input("lib.rs")
        .crate_type("lib")
        .incremental("incr")
        .arg("-Zincremental-verify-ich")
        .extern_("macros", &macros_dylib)
        .run();

    out.assert_stderr_not_contains("internal compiler error");
    out.assert_stderr_not_contains("Found unstable fingerprints");
}

fn find_proc_macro_dylib(name: &str) -> PathBuf {
    let prefix = if cfg!(target_os = "windows") { "" } else { "lib" };

    let ext: &str = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "aix") {
        "a"
    } else {
        "so"
    };

    let lib_name = format!("{prefix}{name}.{ext}");

    for entry in fs::read_dir(".").unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_str().unwrap();
        if name == lib_name {
            return entry.path();
        }
    }

    panic!("could not find proc-macro dylib for `{name}`");
}
