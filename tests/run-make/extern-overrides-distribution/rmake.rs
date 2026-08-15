// The --extern flag should override any "crate_type" declarations in the
// Rust files themselves. In this test, libc is compiled as "lib", but
// main.rs will only run with an rlib, which checks if the --extern flag
// is successfully overriding the default behaviour.
// See https://github.com/rust-lang/rust/pull/21782

//@ ignore-cross-compile

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc().input("libc.rs").metadata("foo").run();
    rustc().input("main.rs").extern_("libc", rust_lib_name("libc")).run();
}
