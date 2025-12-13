//@ only-x86_64-unknown-linux-gnu

use std::fs;
use std::path::Path;

use run_make_support::{cwd, has_extension, llvm_ar, llvm_bcanalyzer, rust_lib_name, rustc};

// A regression test for #146133.

fn main() {
    // Compile a `#![no_builtins]` rlib crate with `-Clinker-plugin-lto`.
    // It is acceptable to generate bitcode for rlib, so there is no need to check something.
    rustc().input("no_builtins.rs").crate_type("rlib").linker_plugin_lto("on").run();

    // Checks that rustc's LTO doesn't emit any bitcode to the linker.
    let stdout = rustc()
        .input("main.rs")
        .extern_("no_builtins", rust_lib_name("no_builtins"))
        .lto("thin")
        .print("link-args")
        .arg("-Csave-temps")
        .arg("-Clinker-features=-lld")
        .run()
        .stdout_utf8();
    for object in stdout
        .split_whitespace()
        .map(|s| s.trim_matches('"'))
        .filter(|path| has_extension(path, "rlib") || has_extension(path, "o"))
    {
        let object_path = if !fs::exists(object).unwrap() {
            cwd().join(object)
        } else {
            Path::new(object).to_path_buf()
        };
        if has_extension(object, "rlib") {
            let ar_stdout = llvm_ar().arg("t").arg(&object_path).run().stdout_utf8();
            llvm_ar().extract().arg(&object_path).run();
            for object in ar_stdout.split_whitespace().filter(|o| has_extension(o, "o")) {
                let object_path = cwd().join(object);
                not_bitcode(&object_path);
            }
        } else {
            not_bitcode(&object_path);
        }
    }
}

fn not_bitcode(object: &Path) {
    llvm_bcanalyzer()
        .input(object)
        .run_fail()
        .assert_stderr_contains("llvm-bcanalyzer: Invalid record at top-level");
}
