// Instead of using the default dlltool, the rust compiler can also accept a custom
// command file with the -C dlltool flag. This test uses it to compile some rust code
// with the raw_dylib Windows-exclusive feature, and checks that the output contains
// the string passed from the custom dlltool, confirming that the default dlltool was
// successfully overridden.
// See https://github.com/rust-lang/rust/pull/109677

//@ only-windows
//@ only-gnu
//@ needs-dlltool
// Reason: this test specifically checks the custom dlltool feature, only
// available on Windows-gnu.

use run_make_support::{diff, rustc};

fn main() {
    let out = rustc()
        .crate_type("lib")
        .crate_name("raw_dylib_test")
        .input("lib.rs")
        .arg("-Cdlltool=script.cmd")
        .run();
    diff().expected_file("output.txt").actual_file("actual.txt").normalize(r#"\r"#, "").run();
}
