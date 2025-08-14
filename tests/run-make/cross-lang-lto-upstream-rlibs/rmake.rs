//@ needs-target-std
//
// When using the flag -C linker-plugin-lto, static libraries could lose their upstream object
// files during compilation. This bug was fixed in #53031, and this test compiles a staticlib
// dependent on upstream, checking that the upstream object file still exists after no LTO and
// thin LTO.
// See https://github.com/rust-lang/rust/pull/53031

use run_make_support::{
    cwd, has_extension, has_prefix, has_suffix, llvm_ar, rfs, rustc, shallow_find_files,
    static_lib_name,
};

fn main() {
    // The test starts with no LTO enabled.
    rustc().input("upstream.rs").arg("-Clinker-plugin-lto").codegen_units(1).run();
    rustc()
        .input("staticlib.rs")
        .arg("-Clinker-plugin-lto")
        .codegen_units(1)
        .output(static_lib_name("staticlib"))
        .run();
    llvm_ar().extract().arg(static_lib_name("staticlib")).run();
    // Ensure the upstream object file was included.
    assert_eq!(
        shallow_find_files(cwd(), |path| {
            has_prefix(path, "upstream.") && has_suffix(path, ".rcgu.o")
        })
        .len(),
        1
    );
    // Remove all output files that are not source Rust code for cleanup.
    for file in shallow_find_files(cwd(), |path| !has_extension(path, "rs")) {
        rfs::remove_file(file)
    }

    // Check it again, with Thin LTO.
    rustc()
        .input("upstream.rs")
        .arg("-Clinker-plugin-lto")
        .codegen_units(1)
        .arg("-Clto=thin")
        .run();
    rustc()
        .input("staticlib.rs")
        .arg("-Clinker-plugin-lto")
        .codegen_units(1)
        .arg("-Clto=thin")
        .output(static_lib_name("staticlib"))
        .run();
    llvm_ar().extract().arg(static_lib_name("staticlib")).run();
    assert_eq!(
        shallow_find_files(cwd(), |path| {
            has_prefix(path, "upstream.") && has_suffix(path, ".rcgu.o")
        })
        .len(),
        1
    );
}
