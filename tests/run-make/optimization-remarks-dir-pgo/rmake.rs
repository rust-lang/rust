// This test checks the -Zremark-dir flag, which writes LLVM
// optimization remarks to the YAML format. When using PGO (Profile
// Guided Optimization), the Hotness attribute should be included in
// the output remark files.
// See https://github.com/rust-lang/rust/pull/114439

//@ needs-profiler-runtime
//@ ignore-cross-compile

use run_make_support::{
    has_extension, has_prefix, invalid_utf8_contains, llvm_profdata, run, rustc, shallow_find_files,
};

fn main() {
    rustc().profile_generate("profdata").opt().input("foo.rs").output("foo").run();
    run("foo");
    // The profdata filename is a long sequence of numbers, fetch it by prefix and extension
    // to keep the test working even if the filename changes.
    let profdata_files = shallow_find_files("profdata", |path| {
        has_prefix(path, "default") && has_extension(path, "profraw")
    });
    let profdata_file = profdata_files.get(0).unwrap();
    llvm_profdata().merge().output("merged.profdata").input(profdata_file).run();
    rustc()
        .profile_use("merged.profdata")
        .opt()
        .input("foo.rs")
        .arg("-Cremark=all")
        .arg("-Zremark-dir=profiles")
        .run();
    // Check that PGO hotness is included in the remark files
    let remark_files = shallow_find_files("profiles", |path| {
        has_prefix(path, "foo") && has_extension(path, "yaml")
    });
    assert!(!remark_files.is_empty());
    for file in remark_files {
        if !file.to_str().unwrap().contains("codegen") {
            invalid_utf8_contains(file, "Hotness")
        };
    }
}
