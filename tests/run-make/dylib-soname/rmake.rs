// Checks that produced dylibs have a relative SONAME set, so they don't put "unmovable" full paths
// into DT_NEEDED when used by a full path.

//@ only-linux
//@ ignore-cross-compile

use run_make_support::regex::Regex;
use run_make_support::{cmd, run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        rustc().crate_name("foo").crate_type("dylib").input("foo.rs").run();
        cmd("readelf")
            .arg("-d")
            .arg("libfoo.so")
            .run()
            .assert_stdout_contains("Library soname: [libfoo.so]");
    });
}
