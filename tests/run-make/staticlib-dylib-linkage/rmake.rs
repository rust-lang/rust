// A basic smoke test to check that rustc supports linking to a rust dylib with
// --crate-type staticlib. bar is a dylib, on which foo is dependent - the native
// static lib search paths are collected and used to compile foo.c, the final executable
// which depends on both foo and bar.
// See https://github.com/rust-lang/rust/pull/106560

//@ ignore-cross-compile
// Reason: the compiled binary is executed.
//@ ignore-wasm
// Reason: WASM does not support dynamic libraries

use run_make_support::{cc, is_windows_msvc, regex, run, rustc, static_lib_name};

fn main() {
    rustc().arg("-Cprefer-dynamic").input("bar.rs").run();
    let libs = rustc()
        .input("foo.rs")
        .crate_type("staticlib")
        .print("native-static-libs")
        .arg("-Zstaticlib-allow-rdylib-deps")
        .run()
        .assert_stderr_contains("note: native-static-libs: ")
        .stderr_utf8();
    let re = regex::Regex::new(r#"note: native-static-libs:\s*(.+)"#).unwrap();
    let libs = re.find(&libs).unwrap().as_str().trim();
    // remove the note
    let (_, native_link_args) = libs.split_once("note: native-static-libs: ").unwrap();
    // divide the command-line arguments in a vec
    let mut native_link_args = native_link_args.split(' ').collect::<Vec<&str>>();
    if is_windows_msvc() {
        // For MSVC pass the arguments on to the linker.
        native_link_args.insert(0, "-link");
    }
    cc().input("foo.c").input(static_lib_name("foo")).args(native_link_args).out_exe("foo").run();
    run("foo");
}
