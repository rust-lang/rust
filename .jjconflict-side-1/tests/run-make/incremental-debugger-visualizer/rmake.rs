//@ needs-target-std
//
// This test ensures that changes to files referenced via #[debugger_visualizer]
// (in this case, foo.py and foo.natvis) are picked up when compiling incrementally.
// See https://github.com/rust-lang/rust/pull/111641

use run_make_support::{invalid_utf8_contains, invalid_utf8_not_contains, rfs, rustc};

fn main() {
    rfs::create_file("foo.py");
    rfs::write("foo.py", "GDB script v1");
    rfs::create_file("foo.natvis");
    rfs::write("foo.natvis", "Natvis v1");
    rustc()
        .input("foo.rs")
        .crate_type("rlib")
        .emit("metadata")
        .incremental("incremental")
        .arg("-Zincremental-verify-ich")
        .run();

    invalid_utf8_contains("libfoo.rmeta", "GDB script v1");
    invalid_utf8_contains("libfoo.rmeta", "Natvis v1");

    // Change only the GDB script and check that the change has been picked up
    rfs::remove_file("foo.py");
    rfs::create_file("foo.py");
    rfs::write("foo.py", "GDB script v2");
    rustc()
        .input("foo.rs")
        .crate_type("rlib")
        .emit("metadata")
        .incremental("incremental")
        .arg("-Zincremental-verify-ich")
        .run();

    invalid_utf8_contains("libfoo.rmeta", "GDB script v2");
    invalid_utf8_not_contains("libfoo.rmeta", "GDB script v1");
    invalid_utf8_contains("libfoo.rmeta", "Natvis v1");

    // Now change the Natvis version and check that the change has been picked up
    rfs::remove_file("foo.natvis");
    rfs::create_file("foo.natvis");
    rfs::write("foo.natvis", "Natvis v2");
    rustc()
        .input("foo.rs")
        .crate_type("rlib")
        .emit("metadata")
        .incremental("incremental")
        .arg("-Zincremental-verify-ich")
        .run();

    invalid_utf8_contains("libfoo.rmeta", "GDB script v2");
    invalid_utf8_not_contains("libfoo.rmeta", "GDB script v1");
    invalid_utf8_not_contains("libfoo.rmeta", "Natvis v1");
    invalid_utf8_contains("libfoo.rmeta", "Natvis v2");
}
