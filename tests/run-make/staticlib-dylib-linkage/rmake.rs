// A basic smoke test to check that rustc supports linking to a rust dylib with
// --crate-type staticlib. bar is a dylib, on which foo is dependent - the native
// static lib search paths are collected and used to compile foo.c, the final executable
// which depends on both foo and bar.
// See https://github.com/rust-lang/rust/pull/106560

//@ ignore-cross-compile
// Reason: the compiled binary is executed.
//@ ignore-wasm
// Reason: WASM does not support dynamic libraries
//@ ignore-msvc
//FIXME(Oneirical): Getting this to work on MSVC requires passing libcmt.lib to CC,
// which is not trivial to do.
// Tracking issue: https://github.com/rust-lang/rust/issues/128602
// Discussion: https://github.com/rust-lang/rust/pull/128407#discussion_r1702439172

use run_make_support::{cc, regex, run, rustc};

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
    let (_, library_search_paths) = libs.split_once("note: native-static-libs: ").unwrap();
    // divide the command-line arguments in a vec
    let library_search_paths = library_search_paths.split(' ').collect::<Vec<&str>>();
    cc().input("foo.c").arg("-lfoo").args(library_search_paths).out_exe("foo").run();
    run("foo");
}
