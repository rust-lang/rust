// This test checks the -Zremark-dir flag, which writes LLVM
// optimization remarks to the YAML format. When using PGO (Profile
// Guided Optimization), the Hotness attribute should be included in
// the output remark files.
// See https://github.com/rust-lang/rust/pull/114439

//@ needs-profiler-support
//@ ignore-cross-compile

use run_make_support::{
    has_extension, has_prefix, invalid_utf8_contains, llvm_profdata, run, rustc, shallow_find_files,
};

fn main() {
    rustc().profile_generate("profdata").opt().input("foo.rs").output("foo").run();
    run("foo");
    llvm_profdata()
        .merge()
        .output("merged.profdata")
        .input(
            shallow_find_files("profdata", |path| {
                has_prefix(path, "default") && has_extension(path, "profraw")
            })
            .get(0)
            .unwrap(),
        )
        .run();
    rustc()
        .profile_use("merged.profdata")
        .opt()
        .input("foo.rs")
        .arg("-Cremark=all")
        .arg("-Zremark-dir=profiles")
        .run();
    // Check that PGO hotness is included in the remark files
    assert!(
        !shallow_find_files("profiles", |path| {
            has_prefix(path, "foo") && has_extension(path, "yaml")
        })
        .is_empty()
    );
    for file in shallow_find_files("profiles", |path| {
        has_prefix(path, "foo") && has_extension(path, "yaml")
    }) {
        if !file.to_str().unwrap().contains("codegen") {
            invalid_utf8_contains(file, "Hotness")
        };
    }
}
