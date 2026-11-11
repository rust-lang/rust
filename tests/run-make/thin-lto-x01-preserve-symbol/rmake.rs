// `\x01` is not part of the ThinLTO GUID, so an optimized build must keep the
// export.

//@ only-elf
//@ ignore-cross-compile
// FIXME: Once GCC backend is fixed, remove this `ignore-backends`.
//@ ignore-backends: gcc

use run_make_support::{llvm_readobj, rustc, static_lib_name};

fn main() {
    // `-O` enables local ThinLTO over the default number of codegen units.
    rustc().input("lib.rs").crate_type("staticlib").opt().run();

    llvm_readobj()
        .arg("--symbols")
        .input(static_lib_name("lib"))
        .run()
        .assert_stdout_contains("ctl_export")
        .assert_stdout_contains("plain_export");
}
