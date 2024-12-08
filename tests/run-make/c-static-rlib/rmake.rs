// This test checks that static Rust linking with C does not encounter any errors,
// with static dependencies given preference over dynamic. (This is the default behaviour.)
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, rfs, run, rust_lib_name, rustc, static_lib_name};

fn main() {
    build_native_static_lib("cfoo");
    rustc().input("foo.rs").run();
    rustc().input("bar.rs").run();
    rfs::remove_file(rust_lib_name("foo"));
    rfs::remove_file(static_lib_name("cfoo"));
    run("bar");
}
