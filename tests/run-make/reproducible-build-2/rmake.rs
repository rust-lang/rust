// Builds with fat link-time-optimizations and the --sysroot flag used to be
// non-deterministic - that means, compiling twice with no changes would create
// slightly different outputs. This has been fixed by #63352 and #63505.
// Test 1: Compile with fat-lto twice, check that both compilation outputs are identical.
// Test 2: Compile with sysroot, then change the sysroot path from absolute to relative.
// Outputs should be identical.
// See https://github.com/rust-lang/rust/issues/34902

//FIXME(Oneirical): excluded ignore-musl ignore-windows ignore-cross-compile

use run_make_support::{fs_wrapper, rust_lib_name, rustc};

fn main() {
    // test 1: fat lto
    rustc().input("reproducible-build-aux.rs").run();
    rustc().input("reproducible-build.rs").arg("-Clto=fat").run();
    fs_wrapper::rename("reproducible-build", "reproducible-build-a");
    rustc().input("reproducible-build.rs").arg("-Clto=fat").run();
    assert_eq!(fs_wrapper::read("reproducible-build"), fs_wrapper::read("reproducible-build-a"));

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
    fs_wrapper::copy_dir_all(&sysroot, "sysroot");
    fs_wrapper::rename(rust_lib_name("reproducible_build"), rust_lib_name("foo"));
    rustc()
        .input("reproducible-build.rs")
        .crate_type("rlib")
        .sysroot("sysroot")
        .arg("--remap-path-prefix=/sysroot=/sysroot")
        .run();

    assert_eq!(
        fs_wrapper::read(rust_lib_name("reproducible_build")),
        fs_wrapper::read(rust_lib_name("foo"))
    );
}
