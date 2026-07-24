use run_make_support::rfs::create_dir_all;
use run_make_support::{rustc, target};

fn main() {
    // Create test sysroot
    let test_sysroot = format!("./testsysroot/lib/rustlib/{}/lib/", target());
    create_dir_all(&test_sysroot);

    // Layout:
    // - Foo depends directly on Bar
    // - Bar depends directly on Baz
    // - Baz can be found in the sysroot.

    // 1) Depending transitively on a lib in the sysroot resolves
    // fine with `-Zimplicit-sysroot-deps=false`
    rustc().input("baz.rs").crate_type("lib").out_dir(test_sysroot).run();
    rustc().input("bar.rs").crate_type("lib").sysroot("./testsysroot").run();
    rustc()
        .input("foo.rs")
        .crate_type("lib")
        .extern_("bar", "libbar.rlib")
        .sysroot("./testsysroot")
        .arg("-Zimplicit-sysroot-deps=false")
        .run();

    // 2) Depending directly on a lib in the sysroot does not resolve
    // implicitly with `-Zimplicit-sysroot-deps=false`
    rustc()
        .input("bar.rs")
        .crate_type("lib")
        .sysroot("./testsysroot")
        .arg("-Zimplicit-sysroot-deps=false")
        .run_fail();
}
