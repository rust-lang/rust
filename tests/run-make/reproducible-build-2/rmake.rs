// Builds with fat link-time-optimizations and the --sysroot flag used to be
// non-deterministic - that means, compiling twice with no changes would create
// slightly different outputs. This has been fixed by #63352 and #63505.
// Test 1: Compile with fat-lto twice, check that both compilation outputs are identical.
// Test 2: Compile with sysroot, then change the sysroot path from absolute to relative.
// Outputs should be identical.
// See https://github.com/rust-lang/rust/issues/34902

//@ needs-target-std
//@ ignore-windows
// Reasons:
// 1. The object files are reproducible, but their paths are not, which causes
// the first assertion in the test to fail.
// 2. When the sysroot gets copied, some symlinks must be re-created,
// which is a privileged action on Windows.

use run_make_support::{rfs, rust_lib_name, rustc};

fn main() {
    // test 1: fat lto
    rustc().input("reproducible-build-aux.rs").run();
    rustc().input("reproducible-build.rs").arg("-Clto=fat").output("reproducible-build").run();
    rfs::rename("reproducible-build", "reproducible-build-a");
    rustc().input("reproducible-build.rs").arg("-Clto=fat").output("reproducible-build").run();
    assert_eq!(rfs::read("reproducible-build"), rfs::read("reproducible-build-a"));

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
