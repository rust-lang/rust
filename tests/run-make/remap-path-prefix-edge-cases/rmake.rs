//! This test checks multiple edge-case of `--remap-path-prefix`.
//!
//! It tests:
//!  - `=` sign in FROM path
//!  - multiple path remappings
//!  - multiple conflicting path remappings

//@ ignore-windows (does not support directories with = sign)

use std::path::Path;

use run_make_support::{
    CompletedProcess, assert_contains, assert_not_contains, cwd, rfs, run_in_tmpdir, rustc, rustdoc,
};

fn main() {
    run_in_tmpdir(|| {
        let out_dir = cwd();

        // Create a directory with an `=` sign
        let eq_dir = out_dir.join("path=with=equal");
        rfs::create_dir_all(&eq_dir);

        let src_path = eq_dir.join("lib.rs");
        rfs::write(&src_path, "pub fn broken_func() { ");

        // Use multiple remap args and conflicting remappings
        let remap_args = [
            format!("--remap-path-prefix={}={}", eq_dir.display(), "REMAPPED_DIR"),
            format!("--remap-path-prefix={}={}", eq_dir.display(), "REMAPPED_DIR2"),
        ];

        fn run_test(cmd: impl FnOnce() -> CompletedProcess) {
            let output = cmd();
            let stderr = output.stderr_utf8();

            // Checks the diagnostic output
            assert_contains(&stderr, "REMAPPED_DIR2/lib.rs");
            assert_not_contains(&stderr, "REMAPPED_DIR/");
            assert_not_contains(&stderr, "path=with=equal");
        };

        // Test with rustc
        run_test(|| rustc().input(&src_path).args(&remap_args).run_fail());

        // Test with rustdoc
        run_test(|| {
            rustdoc().input(&src_path).arg("-Zunstable-options").args(&remap_args).run_fail()
        });
    });
}
