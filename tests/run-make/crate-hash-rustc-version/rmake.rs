// Ensure that crates compiled with different rustc versions cannot
// be dynamically linked.

//@ ignore-cross-compile
//@ only-unix

use run_make_support::{cmd, diff, dynamic_lib_name, is_darwin, run, run_fail, rustc};

fn main() {
    let flags = ["-Cprefer-dynamic", "-Csymbol-mangling-version=v0"];
    let nm_flag = if is_darwin() { [].as_slice() } else { ["-D"].as_slice() };

    // a.rs is compiled to a dylib
    rustc().input("a.rs").crate_type("dylib").args(&flags).run();

    // Store symbols
    let symbols_before = cmd("nm").args(nm_flag).arg(dynamic_lib_name("a")).run().stdout_utf8();

    // b.rs is compiled to a binary
    rustc()
        .input("b.rs")
        .extern_("a", dynamic_lib_name("a"))
        .crate_type("bin")
        .arg("-Crpath")
        .args(&flags)
        .run();
    run("b");

    // Now re-compile a.rs with another rustc version
    rustc()
        .env("RUSTC_FORCE_RUSTC_VERSION", "deadfeed")
        .input("a.rs")
        .crate_type("dylib")
        .args(&flags)
        .run();

    // After compiling with a different rustc version, store symbols again.
    let symbols_after = cmd("nm").args(nm_flag).arg(dynamic_lib_name("a")).run().stdout_utf8();

    // As a sanity check, test if the symbols changed:
    // If the symbols are identical, there's been an error.
    diff()
        .expected_text("symbols_before", symbols_before)
        .actual_text("symbols_after", symbols_after)
        .run_fail();
    run_fail("b");
}
