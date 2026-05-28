// The gcc driver is supposed to add a terminator to link files, and the rustc
// driver previously failed to do this, resulting in a segmentation fault
// with an older version of LLVM. This test checks that the terminator is present
// after the fix in #85395.
// See https://github.com/rust-lang/rust/issues/47551

//@ only-linux
// Reason: the ZERO terminator is unique to the Linux architecture.
//@ ignore-32bit
// Reason: the usage of a large array in the test causes an out-of-memory
// error on 32 bit systems.
//@ ignore-cross-compile

use run_make_support::{bin_name, llvm_objdump, run, rustc};

fn main() {
    rustc().input("eh_frame-terminator.rs").run();
    run("eh_frame-terminator").assert_stdout_contains("1122334455667788");
    llvm_objdump()
        .arg("--dwarf=frames")
        .input(bin_name("eh_frame-terminator"))
        .run()
        .assert_stdout_contains("ZERO terminator");
}
