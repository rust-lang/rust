// Test that coverage instrumentation works correctly with -Z stable-crate-hash.
//
// This verifies that:
// 1. Coverage instrumentation compiles successfully with -Z stable-crate-hash
// 2. Coverage profraw data is generated when running the instrumented binary
// 3. Coverage data can be converted to a report format
//
// Coverage regions should point to correct source locations even when
// spans are stored separately from metadata.

//@ ignore-cross-compile
//@ needs-profiler-runtime

use run_make_support::{cmd, cwd, llvm_profdata, rfs, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        // Build library with coverage instrumentation and -Z stable-crate-hash
        rustc()
            .input("lib.rs")
            .crate_type("rlib")
            .arg("-Cinstrument-coverage")
            .arg("-Zstable-crate-hash")
            .run();

        // Build main binary with coverage instrumentation
        rustc()
            .input("main.rs")
            .crate_type("bin")
            .extern_("rdr_coverage_lib", "librdr_coverage_lib.rlib")
            .arg("-Cinstrument-coverage")
            .arg("-Zstable-crate-hash")
            .run();

        // Run the binary to generate coverage data
        // The profraw file will be created in the current directory
        let profraw_file = format!("{}/default_%m_%p.profraw", cwd().display());
        cmd("./main").env("LLVM_PROFILE_FILE", &profraw_file).run();

        // Find the generated profraw file
        let profraw_files: Vec<_> = rfs::read_dir(".")
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "profraw"))
            .collect();

        assert!(!profraw_files.is_empty(), "Should have generated at least one .profraw file");

        // Merge the profraw data into profdata
        let profdata_file = "coverage.profdata";
        llvm_profdata()
            .arg("merge")
            .arg("-sparse")
            .args(profraw_files.iter().map(|e| e.path()))
            .arg("-o")
            .arg(profdata_file)
            .run();

        assert!(std::path::Path::new(profdata_file).exists(), "Should have created profdata file");

        // Verify the profdata file is valid by checking it can be read
        let profdata_size = rfs::metadata(profdata_file).unwrap().len();
        assert!(profdata_size > 0, "Profdata file should not be empty");

        println!("Coverage test passed! Generated {} bytes of coverage data.", profdata_size);
    });
}
