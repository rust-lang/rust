// `-Z packed_bundled_libs` is an unstable rustc flag that makes the compiler
// only require a native library and no supplementary object files to compile.
// #105601 made it possible to have this behaviour without an unstable flag by
// passing +bundle in modifiers, and this test checks that this feature successfully
// compiles and includes only the static libraries, with no object files.
// See https://github.com/rust-lang/rust/pull/105601

use run_make_support::{
    build_native_static_lib, llvm_ar, regex, rfs, rust_lib_name, rustc, static_lib_name,
};

//@ ignore-cross-compile
// Reason: Invalid library format (not ELF) causes compilation failure
// in the final `rustc` call.

//@ only-linux
// Reason: differences in the native lib compilation process causes differences
// in the --print link-args output

fn main() {
    build_native_static_lib("native_dep_1");
    build_native_static_lib("native_dep_2");
    build_native_static_lib("native_dep_3");
    build_native_static_lib("native_dep_4");
    // Test cfg with packed bundle.
    rustc().input("rust_dep_cfg.rs").crate_type("rlib").run();
    rustc()
        .input("main.rs")
        .extern_("rust_dep", rust_lib_name("rust_dep_cfg"))
        .crate_type("staticlib")
        .cfg("should_add")
        .run();
    // Only static libraries should appear, no object files at all.
    llvm_ar()
        .arg("t")
        .arg(rust_lib_name("rust_dep_cfg"))
        .run()
        .assert_stdout_contains(static_lib_name("native_dep_1"));
    llvm_ar()
        .arg("t")
        .arg(rust_lib_name("rust_dep_cfg"))
        .run()
        .assert_stdout_contains(static_lib_name("native_dep_2"));
    llvm_ar().arg("t").arg(static_lib_name("main")).run().assert_stdout_contains("native_dep_1.o");
    llvm_ar()
        .arg("t")
        .arg(static_lib_name("main"))
        .run()
        .assert_stdout_not_contains("native_dep_2.o");

    // Test bundle with whole archive.
    rustc().input("rust_dep.rs").crate_type("rlib").run();
    // Only deps with `+bundle` should appear.
    llvm_ar().arg("t").arg(rust_lib_name("rust_dep")).run().assert_stdout_contains("native_dep_1");
    llvm_ar().arg("t").arg(rust_lib_name("rust_dep")).run().assert_stdout_contains("native_dep_3");
    llvm_ar()
        .arg("t")
        .arg(rust_lib_name("rust_dep"))
        .run()
        .assert_stdout_not_contains("native_dep_2");
    llvm_ar()
        .arg("t")
        .arg(rust_lib_name("rust_dep"))
        .run()
        .assert_stdout_not_contains("native_dep_4");

    // The compiler shouldn't use files which it doesn't know about.
    rfs::remove_file(static_lib_name("native_dep_1"));
    rfs::remove_file(static_lib_name("native_dep_3"));

    let out = rustc()
        .input("main.rs")
        .extern_("rust_dep", rust_lib_name("rust_dep"))
        .print("link-args")
        .run()
        .assert_stdout_not_contains("native_dep_3")
        .stdout_utf8();

    let re = regex::Regex::new(
"--whole-archive.*native_dep_1.*--whole-archive.*lnative_dep_2.*no-whole-archive.*lnative_dep_4"
    ).unwrap();

    assert!(re.is_match(&out));
}
