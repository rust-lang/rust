// This test makes sure that changes to files referenced via //[debugger_visualizer]
// are picked up when compiling incrementally.

// We have to copy the source to $(TMPDIR) because Github CI mounts the source
// directory as readonly. We need to apply modifications to some of the source
// file.

use run_make_support::{
    fs_wrapper, invalid_utf8_contains_str, invalid_utf8_not_contains_str, rustc,
};
use std::io::Read;

fn main() {
    fs_wrapper::create_file("foo.py");
    fs_wrapper::write("foo.py", "GDB script v1");
    fs_wrapper::create_file("foo.natvis");
    fs_wrapper::write("foo.py", "Natvis v1");
    rustc()
        .input("foo.rs")
        .crate_type("rlib")
        .emit("metadata")
        .incremental("incremental")
        .arg("-Zincremental-verify-ich")
        .run();

    invalid_utf8_contains_str("libfoo.rmeta", "GDB script v1");
    invalid_utf8_contains_str("libfoo.rmeta", "Natvis v1");

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

    invalid_utf8_contains_str("libfoo.rmeta", "GDB script v2");
    invalid_utf8_not_contains_str("libfoo.rmeta", "GDB script v1");
    invalid_utf8_contains_str("libfoo.rmeta", "Natvis v1");

    // Now change the Natvis version and check that the change has been picked up
    fs_wrapper::remove_file("foo.natvis");
    fs_wrapper::create_file("foo.natvis");
    fs_wrapper::write("foo.py", "Natvis v2");
    rustc()
        .input("foo.rs")
        .crate_type("rlib")
        .emit("metadata")
        .incremental("incremental")
        .arg("-Zincremental-verify-ich")
        .run();

    invalid_utf8_contains_str("libfoo.rmeta", "GDB script v2");
    invalid_utf8_not_contains_str("libfoo.rmeta", "GDB script v1");
    invalid_utf8_not_contains_str("libfoo.rmeta", "Natvis v1");
    invalid_utf8_contains_str("libfoo.rmeta", "Natvis v2");
}
