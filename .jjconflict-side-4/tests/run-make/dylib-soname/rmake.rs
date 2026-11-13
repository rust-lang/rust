// Checks that produced dylibs have a relative SONAME set, so they don't put "unmovable" full paths
// into DT_NEEDED when used by a full path.

//@ only-linux
//@ ignore-cross-compile

use run_make_support::{cmd, run_in_tmpdir, rustc};

fn main() {
    let check = |ty: &str| {
        rustc().crate_name("foo").crate_type(ty).input("foo.rs").run();
        cmd("readelf").arg("-d").arg("libfoo.so").run()
    };
    run_in_tmpdir(|| {
        // Rust dylibs should get a relative SONAME
        check("dylib").assert_stdout_contains("Library soname: [libfoo.so]");
    });
    run_in_tmpdir(|| {
        // C dylibs should not implicitly get any SONAME
        check("cdylib").assert_stdout_not_contains("Library soname:");
    });
}
