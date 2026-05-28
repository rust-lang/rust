//@ needs-target-std
//
// A similar test to pass-linker-flags, testing that the `-l link-arg` flag
// respects the order relative to other `-l` flags, but this time, the flags
// are passed on the compilation of a dependency. This test checks that the
// downstream compiled binary contains the linker arguments of the dependency,
// and in the correct order.
// See https://github.com/rust-lang/rust/issues/99427

use run_make_support::{regex, rust_lib_name, rustc};

fn main() {
    // Build dependencies
    rustc().input("native_dep_1.rs").crate_type("staticlib").run();
    rustc().input("native_dep_2.rs").crate_type("staticlib").run();
    rustc()
        .input("rust_dep_flag.rs")
        .arg("-lstatic:-bundle=native_dep_1")
        .arg("-llink-arg=some_flag")
        .arg("-lstatic:-bundle=native_dep_2")
        .crate_type("lib")
        .arg("-Zunstable-options")
        .run();
    rustc().input("rust_dep_attr.rs").crate_type("lib").run();

    // Check sequence of linker arguments
    let out_flag = rustc()
        .input("main.rs")
        .extern_("lib", rust_lib_name("rust_dep_flag"))
        .crate_type("bin")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();
    let out_attr = rustc()
        .input("main.rs")
        .extern_("lib", rust_lib_name("rust_dep_attr"))
        .crate_type("bin")
        .print("link-args")
        .run_unchecked()
        .stdout_utf8();

    let re = regex::Regex::new("native_dep_1.*some_flag.*native_dep_2").unwrap();
    assert!(re.is_match(&out_flag));
    assert!(re.is_match(&out_attr));
}
