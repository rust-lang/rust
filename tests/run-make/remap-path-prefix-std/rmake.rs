// This test makes sure that we do not leak paths to the checkout
// (ie. /checkout in CI) in the distributed standard library debuginfo.
// It checks all rlibs found in the target libdir, not just libstd.
//
// This test only runs on Linux and dist builder (or with `rust.remap-debuginfo = true`
// set in your `bootstrap.toml`).

//@ needs-std-remap-debuginfo
//@ only-linux

use std::path::PathBuf;

use run_make_support::{llvm_dwarfdump, rfs, rustc, shallow_find_files, source_root};

fn main() {
    // Find the target libdir for the current target
    let target_libdir = {
        let output = rustc().print("target-libdir").run();
        let stdout = output.stdout_utf8();
        let path = PathBuf::from(stdout.trim());

        // Assert that the target-libdir path exists
        assert!(path.exists(), "target-libdir: {path:?} does not exists");

        path
    };

    // Find all rlib files under the libdir (the full standard library set)
    let all_rlibs = shallow_find_files(&target_libdir, |p| {
        if let Some(filename) = p.file_name()
            && let filename = filename.to_string_lossy()
            && let Some(ext) = p.extension()
            && filename.starts_with("lib")
            && ext == "rlib"
        {
            true
        } else {
            false
        }
    });

    // There must be at least one rlib (libstd itself, plus many others)
    assert!(!all_rlibs.is_empty(), "no rlibs found in target libdir {target_libdir:?}");

    for rlib in &all_rlibs {
        // Use a stable symlink name based on the crate part (before the '-<hash>' suffix).
        // e.g. "libstd-92abaa9b58c011c1.rlib" → "libstd.rlib"
        let filename = rlib.file_name().unwrap().to_string_lossy();
        let link_name = match filename.split_once('-') {
            Some((prefix, _)) => format!("{prefix}.rlib"),
            None => filename.to_string(),
        };

        // Symlink the original rlib to avoid absolute paths from dwarfdump itself
        rfs::symlink_file(rlib, &link_name);

        // Check that no distributed rlib leaks the checkout/source root path.
        let completed = llvm_dwarfdump()
            .input(&link_name)
            .run()
            .assert_stdout_not_contains(source_root().to_string_lossy());

        // Check that we have `/rustc` in the output if the rlib has any debug info.
        if completed.stdout_utf8().contains("DW_TAG_compile_unit") {
            completed.assert_stdout_contains("/rustc");
        }
    }
}
