//! Tests that deployment target linker warnings are shown as `linker-info`, not `linker-messages`

//@ only-macos

use run_make_support::external_deps::c_cxx_compiler::cc;
use run_make_support::external_deps::llvm::llvm_ar;
use run_make_support::{host, rustc, diff};

fn main() {
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
        .normalize(r"\(.*/rmake_out/", "(TEST_DIR/")
        .run()
}
