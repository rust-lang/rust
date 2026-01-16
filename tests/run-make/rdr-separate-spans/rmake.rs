// Test that -Z separate-spans flag produces reproducible metadata.
//
// This test verifies:
// 1. The -Z separate-spans flag is accepted by the compiler
// 2. Compilation produces a .spans file alongside the .rmeta file (when implemented)
// 3. The resulting rlib is reproducible across builds from different directories
//
// See https://github.com/rust-lang/rust/issues/XXXXX for the RDR tracking issue.

//@ ignore-cross-compile

use run_make_support::{cwd, rfs, run_in_tmpdir, rust_lib_name, rustc};

fn main() {
    // Test 1: Basic compilation with -Z separate-spans succeeds
    run_in_tmpdir(|| {
        rustc().input("lib.rs").crate_type("rlib").arg("-Zseparate-spans").run();

        // Verify the rlib was created
        assert!(
            std::path::Path::new(&rust_lib_name("rdr_test_lib")).exists(),
            "rlib should be created with -Z separate-spans"
        );
    });

    // Test 2: Reproducibility - same source compiled twice should produce identical rlibs
    run_in_tmpdir(|| {
        // First compilation
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Zseparate-spans")
            .arg(&format!("--remap-path-prefix={}=/src", cwd().display()))
            .run();

        let first_rlib = rfs::read(rust_lib_name("rdr_test_lib"));
        rfs::rename(rust_lib_name("rdr_test_lib"), "first.rlib");

        // Second compilation (identical)
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Zseparate-spans")
            .arg(&format!("--remap-path-prefix={}=/src", cwd().display()))
            .run();

        let second_rlib = rfs::read(rust_lib_name("rdr_test_lib"));

        assert_eq!(
            first_rlib, second_rlib,
            "Two identical compilations with -Z separate-spans should produce identical rlibs"
        );
    });

    // Test 3: Compilation from different directories should produce identical rlibs
    // when using appropriate path remapping
    run_in_tmpdir(|| {
        let base_dir = cwd();

        // Create subdirectory with copy of source
        rfs::create_dir("subdir");
        rfs::copy("lib.rs", "subdir/lib.rs");

        // Compile from base directory
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Zseparate-spans")
            .arg(&format!("--remap-path-prefix={}=/src", base_dir.display()))
            .run();

        let base_rlib = rfs::read(rust_lib_name("rdr_test_lib"));
        rfs::rename(rust_lib_name("rdr_test_lib"), "base.rlib");

        // Compile from subdirectory
        std::env::set_current_dir("subdir").unwrap();
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Zseparate-spans")
            .arg(&format!("--remap-path-prefix={}=/src", base_dir.join("subdir").display()))
            .out_dir(&base_dir)
            .run();

        std::env::set_current_dir(&base_dir).unwrap();
        let subdir_rlib = rfs::read(rust_lib_name("rdr_test_lib"));

        assert_eq!(
            base_rlib, subdir_rlib,
            "Compilation from different directories should produce identical rlibs"
        );
    });
}
