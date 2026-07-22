// ignore-tidy-file-linelength
//! Tests that deployment target linker warnings are shown as `linker-info`, not `linker-messages`
//! See <https://github.com/rust-lang/rust/issues/156714>

//@ only-macos

use run_make_support::external_deps::c_cxx_compiler::cc;
use run_make_support::external_deps::llvm::llvm_ar;
use run_make_support::{diff, rustc};

fn main() {
    let ld64_obj = r"ld: warning: object file \(.*\) was built for newer .+ version \(\d+\.\d+\) than being linked \(\d+\.\d+\)";
    let ld_prime_obj = r"ld: warning: object file \(.*\) was built for newer '.+' version \(\d+\.\d+\) than being linked \(\d+\.\d+\)";
    let ld64_dylib = r"ld: warning: dylib \(.*\) was built for newer .+ version \(\d+\.\d+\) than being linked \(\d+\.\d+\)";
    let ld_prime_dylib = r"ld: warning: building for [^ ,]+, but linking with dylib '[^']*' which was built for newer version [0-9.]+";

    // Test 1: static archive (object file mismatch)
    cc().arg("-c").arg("-mmacosx-version-min=15.5").output("foo.o").input("foo.c").run();
    llvm_ar().obj_to_ar().output_input("libfoo.a", "foo.o").run();

    let warnings = rustc()
        .arg("-lstatic=foo")
        .link_arg("-mmacosx-version-min=11.2")
        .input("main.rs")
        .crate_type("bin")
        .run()
        .stderr_utf8();

    diff()
        .expected_file("warnings.txt")
        .actual_text("(rustc -W linker-info)", &warnings)
        .normalize(ld64_obj, "NORMALIZED_OBJECT_DEPLOYMENT_MISMATCH_LINKER_WARNING")
        .normalize(ld_prime_obj, "NORMALIZED_OBJECT_DEPLOYMENT_MISMATCH_LINKER_WARNING")
        .run();

    // Test 2: shared library (dylib mismatch)
    cc().arg("-shared")
        .arg("-mmacosx-version-min=15.5")
        .output("libbar.dylib")
        .input("foo.c")
        .run();

    let dylib_warnings = rustc()
        .arg("-lbar")
        .link_arg("-mmacosx-version-min=11.2")
        .input("main_dylib.rs")
        .crate_type("bin")
        .run()
        .stderr_utf8();

    diff()
        .expected_file("dylib_warnings.txt")
        .actual_text("(rustc -W linker-info dylib)", &dylib_warnings)
        .normalize(ld64_dylib, "NORMALIZED_DYLIB_DEPLOYMENT_MISMATCH_LINKER_WARNING")
        .normalize(ld_prime_dylib, "NORMALIZED_DYLIB_DEPLOYMENT_MISMATCH_LINKER_WARNING")
        .run();
}
