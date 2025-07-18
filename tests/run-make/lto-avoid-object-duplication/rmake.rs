// ignore-tidy-tab
// Staticlibs don't include Rust object files from upstream crates if the same
// code was already pulled into the lib via LTO. However, the bug described in
// https://github.com/rust-lang/rust/issues/64153 lead to this exclusion not
// working properly if the upstream crate was compiled with an explicit filename
// (via `-o`).

// This test makes sure that functions defined in the upstream crates do not
// appear twice in the final staticlib when listing all the symbols from it.

//@ needs-target-std
//@ ignore-windows
// Reason: `llvm-objdump`'s output looks different on windows than on other platforms.
// Only checking on Unix platforms should suffice.
//FIXME(Oneirical): This could be adapted to work on Windows by checking how
// that output differs.

use run_make_support::{llvm_objdump, regex, rust_lib_name, rustc, static_lib_name};

fn main() {
    rustc()
        .crate_type("rlib")
        .input("upstream.rs")
        .output(rust_lib_name("upstream"))
        .codegen_units(1)
        .run();
    rustc()
        .crate_type("staticlib")
        .input("downstream.rs")
        .arg("-Clto")
        .output(static_lib_name("downstream"))
        .codegen_units(1)
        .run();
    let syms = llvm_objdump().arg("-t").input(static_lib_name("downstream")).run().stdout_utf8();
    let re = regex::Regex::new(r#"\s*g\s*F\s.*issue64153_test_function"#).unwrap();
    // Count the global instances of `issue64153_test_function`. There'll be 2
    // if the `upstream` object file got erroneously included twice.
    // The line we are testing for with the regex looks something like:
    // 0000000000000000 g     F .text.issue64153_test_function	00000023 issue64153_test_function
    assert_eq!(re.find_iter(syms.as_str()).count(), 1);
}
