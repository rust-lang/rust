// This test ensures that changes to files referenced via #[debugger_visualizer]
// (in this case, foo.py and foo.natvis) are picked up when compiling incrementally.
// See https://github.com/rust-lang/rust/pull/111641

use run_make_support::{fs_wrapper, invalid_utf8_contains, invalid_utf8_not_contains, rustc};
use std::io::Read;

fn main() {
    fs_wrapper::create_file("foo.py");
    fs_wrapper::write("foo.py", "GDB script v1");
    fs_wrapper::create_file("foo.natvis");
    fs_wrapper::write("foo.natvis", "Natvis v1");
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
    fs_wrapper::remove_file("foo.py");
    fs_wrapper::create_file("foo.py");
    fs_wrapper::write("foo.py", "GDB script v2");
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
    fs_wrapper::remove_file("foo.natvis");
    fs_wrapper::create_file("foo.natvis");
    fs_wrapper::write("foo.natvis", "Natvis v2");
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
