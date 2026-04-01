// This test is designed to intentionally introduce a circular dependency scenario to check
// that a specific compiler bug doesn't make a resurgence.
// The bug in question arose when at least one crate
// required a global allocator, and that crate was placed after
// the one defining it in the linker order.
// The generated symbols.o should not result in any linker errors.
// See https://github.com/rust-lang/rust/issues/112715

//@ ignore-cross-compile

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().input("my_lib.rs").run();
    rustc().input("main.rs").arg("--test").extern_("my_lib", rust_lib_name("my_lib")).run();
}
