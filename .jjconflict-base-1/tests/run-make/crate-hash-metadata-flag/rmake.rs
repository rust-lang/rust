// `-Z metadata-crate-hash=no` is the safety fallback that reverts the crate hash (SVH)
// computation from the encoded crate metadata back to the legacy HIR-based scheme. This test
// checks that the flag actually changes the SVH for an otherwise identical crate, and that each
// mode is deterministic.

//@ ignore-cross-compile

use run_make_support::{diff, rfs, rustc};

/// Build `foo.rs` into `dir` (optionally with the legacy hashing flag) and return the
/// `-Zls=root` metadata dump, which includes the crate hash (SVH).
fn build_in(dir: &str, metadata_crate_hash: bool) -> String {
    let mut cmd = rustc();
    cmd.input("foo.rs").crate_type("rlib").out_dir(dir);
    if !metadata_crate_hash {
        cmd.arg("-Zmetadata-crate-hash=no");
    }
    cmd.run();
    rustc().arg("-Zls=root").input(format!("{dir}/libfoo.rlib")).run().stdout_utf8()
}

fn main() {
    rfs::create_dir("default");
    rfs::create_dir("legacy");
    rfs::create_dir("default_again");
    rfs::create_dir("legacy_again");

    let default = build_in("default", true);
    let legacy = build_in("legacy", false);

    // The SVH (printed by `-Zls=root`) must differ between the metadata-based default and the
    // legacy HIR-based scheme.
    diff().expected_text("default", &default).actual_text("legacy", &legacy).run_fail();

    // Each mode must be deterministic.
    let default_again = build_in("default_again", true);
    diff().expected_text("default", &default).actual_text("default_again", default_again).run();

    let legacy_again = build_in("legacy_again", false);
    diff().expected_text("legacy", &legacy).actual_text("legacy_again", legacy_again).run();
}
