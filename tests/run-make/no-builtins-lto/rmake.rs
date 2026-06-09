//@ ignore-cross-compile

// The rlib produced by a no_builtins crate should be explicitly linked
// during compilation, and as a result be present in the linker arguments.
// See the comments inside this file for more details.
// See https://github.com/rust-lang/rust/pull/35637

use run_make_support::{rust_lib_name, rustc};

fn main() {
    // Compile a `#![no_builtins]` rlib crate
    rustc().input("no_builtins.rs").run();
    // Build an executable that depends on that crate using LTO. The no_builtins crate doesn't
    // participate in LTO, so its rlib must be explicitly
    // linked into the final binary. Verify this by grepping the linker arguments.
    rustc()
        .input("main.rs")
        .arg("-Clto")
        .print("link-args")
        .run()
        .assert_stdout_contains(rust_lib_name("no_builtins"));
}
