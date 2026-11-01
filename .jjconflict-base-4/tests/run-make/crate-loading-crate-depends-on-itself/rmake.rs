//@ only-linux
//@ ignore-wasm32
//@ ignore-wasm64
// ignore-tidy-linelength

// Verify that if the current crate depends on a different version of the same crate, *and* types
// and traits of the different versions are mixed, we produce diagnostic output and not an ICE.
// #133563

use run_make_support::{diff, rust_lib_name, rustc};

fn main() {
    rustc().input("foo-prev.rs").run();

    let out = rustc()
        .extra_filename("current")
        .metadata("current")
        .input("foo-current.rs")
        .extern_("foo", rust_lib_name("foo"))
        .run_fail()
        .stderr_utf8();

    // We don't remap the path of the `foo-prev` crate, so we remap it here.
    let mut lines: Vec<_> = out.lines().collect();
    for line in &mut lines {
        if line.starts_with("  ::: ") {
            *line = "  ::: foo-prev.rs:X:Y";
        }
    }
    diff().expected_file("foo.stderr").actual_text("(rustc)", &lines.join("\n")).run();
}
