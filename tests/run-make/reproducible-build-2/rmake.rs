// Builds with fat link-time-optimizations and the --sysroot flag used to be
// non-deterministic - that means, compiling twice with no changes would create
// slightly different outputs. This has been fixed by #63352 and #63505.
// Test 1: Compile with fat-lto twice, check that both compilation outputs are identical.
// Test 2: Compile with sysroot, then change the sysroot path from absolute to relative.
// Outputs should be identical.
// See https://github.com/rust-lang/rust/issues/34902

//@ ignore-cross-compile

//@ ignore-windows-gnu
// GNU Linker for Windows is non-deterministic.

use run_make_support::{bin_name, is_windows_msvc, rfs, rust_lib_name, rustc};

fn main() {
    // test 1: fat lto
    rustc().input("reproducible-build-aux.rs").run();
    let make_reproducible_build = || {
        let mut reproducible_build = rustc();
        reproducible_build
            .input("reproducible-build.rs")
            .arg("-Clto=fat")
            .output(bin_name("reproducible-build"));
        if is_windows_msvc() {
            // Avoids timestamps, etc. when linking.
            reproducible_build.arg("-Clink-arg=/Brepro");
        }
        reproducible_build.run();
    };
    make_reproducible_build();
    rfs::rename(bin_name("reproducible-build"), "reproducible-build-a");
    if is_windows_msvc() {
        // Linker acts differently if there is already a PDB file with the same
        // name.
        rfs::remove_file("reproducible-build.pdb");
    }
    make_reproducible_build();
    assert_eq!(rfs::read(bin_name("reproducible-build")), rfs::read("reproducible-build-a"));

    // test 2: sysroot
    let sysroot = rustc().print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();

    rustc().input("reproducible-build-aux.rs").run();
    rustc()
        .input("reproducible-build.rs")
        .crate_type("rlib")
        .sysroot(&sysroot)
        .arg(format!("--remap-path-prefix={sysroot}=/sysroot"))
        .run();
    rfs::copy_dir_all(&sysroot, "sysroot");
    rfs::rename(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
    rustc()
        .input("reproducible-build.rs")
        .crate_type("rlib")
        .sysroot("sysroot")
        .arg("--remap-path-prefix=/sysroot=/sysroot")
        .run();

    assert_eq!(rfs::read(rust_lib_name("reproducible_build")), rfs::read(rust_lib_name("foo")));
}
