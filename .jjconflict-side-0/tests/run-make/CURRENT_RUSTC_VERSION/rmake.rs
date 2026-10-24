//@ needs-target-std
// ignore-tidy-linelength

// Check that the `CURRENT_RUSTC_VERSION` placeholder is correctly replaced by the current
// `rustc` version and the `since` property in feature stability gating is properly respected.

use run_make_support::{rfs, rustc, source_root};

fn main() {
    rustc().crate_type("lib").input("stable.rs").emit("metadata").run();

    let output =
        rustc().input("main.rs").emit("metadata").extern_("stable", "libstable.rmeta").run();
    let version = rfs::read_to_string(source_root().join("src/version"));
    let expected_string = format!("stable since {}", version.trim());
    output.assert_stderr_contains(expected_string);
}
