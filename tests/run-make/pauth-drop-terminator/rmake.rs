// Make sure that for `aarch64-unknown-linux-pauthtest` compiler correctly signs drop terminators.
// Please note that the generated pattern:
// ```llvm
//   tail call void ptrauth (ptr @c_cleanup, i32 0)(ptr @c_cleanup, i32 0, i64 2712) #2 [ "ptrauth"(i32 0, i64 2712) ]
// ```
// is optimised out by LLVM's instcombine, hence dump the IR before that pass and inspect it.

//@ only-pauthtest
// ignore-tidy-file-linelength

use run_make_support::path_helpers::source_root;
use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    let sibling = source_root().join("tests/run-make/pauth-drop-terminator");

    let output = rustc()
        .input("main.rs")
        .target("aarch64-unknown-linux-pauthtest")
        .opt_level("3")
        .arg("--crate-type=lib")
        .arg("--emit=llvm-ir")
        .arg("-C")
        .arg("llvm-args=-print-before=instcombine")
        .run();

    let stderr = output.stderr_utf8();

    // -print-before outputs to stderr, so copy it over to a file, that can later be used by
    // filecheck.
    rfs::write("before_instcombine.ll", stderr);

    llvm_filecheck()
        .patterns(sibling.join("before_instcombine.check"))
        .stdin_buf(rfs::read("before_instcombine.ll"))
        .run();

    llvm_filecheck().patterns(sibling.join("full_ir.check")).stdin_buf(rfs::read("main.ll")).run();

    // Compile again now using function pointer type discrimination.
    let output = rustc()
        .input("main.rs")
        .target("aarch64-unknown-linux-pauthtest")
        .opt_level("3")
        .arg("--crate-type=lib")
        .arg("--emit=llvm-ir")
        .arg("-Zpointer-authentication=+function-pointer-type-discrimination")
        .arg("-Cunsafe-allow-abi-mismatch=pointer-authentication")
        .arg("-C")
        .arg("llvm-args=-print-before=instcombine")
        .run();

    let stderr = output.stderr_utf8();

    rfs::write("before_instcombine_ty_disc.ll", stderr);

    llvm_filecheck()
        .patterns(sibling.join("before_instcombine_ty_disc.check"))
        .stdin_buf(rfs::read("before_instcombine_ty_disc.ll"))
        .run();
}
