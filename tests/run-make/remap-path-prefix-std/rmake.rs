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
            && filename.contains('-')
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
        let completed = llvm_dwarfdump().input(&link_name).run_unchecked();
        if !completed.status().success() {
            eprintln!("dwarfdump failed on {link_name}: exit status {:?}", completed.status());
            panic!("llvm-dwarfdump failed for {link_name}");
        }

        let stdout = completed.stdout_utf8();
        let source_root = source_root();
        let root = source_root.to_string_lossy();

        if let Some((i, _)) =
            stdout.lines().enumerate().find(|(_, line)| line.contains(root.as_ref()))
        {
            let lines: Vec<_> = stdout.lines().collect();

            let start = i.saturating_sub(2);
            let end = (i + 3).min(lines.len());

            eprintln!("leaked source-root path found in {link_name}:");

            for line in &lines[start..end] {
                eprintln!("{line}");
            }
        }

        // Check that remapped paths are present if the rlib has debug info.
        if stdout.contains("DW_TAG_compile_unit") {
            assert!(
                stdout.contains("/rustc/") || stdout.contains("/rust/deps"),
                "Expected remapped paths in dwarfdump output for {link_name}",
            );
        }
    }
}
