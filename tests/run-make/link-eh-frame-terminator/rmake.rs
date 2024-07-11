// The gcc driver is supposed to add a terminator to link files, and the rustc
// driver previously failed to do this, resulting in a segmentation fault
// with an older version of LLVM. This test checks that the terminator is present
// after the fix in #85395.
// See https://github.com/rust-lang/rust/issues/47551

//FIXME(Oneirical): See if it works on anything other than only linux and 64 bit
// maybe riscv64gc-unknown-linux-gnu

use run_make_support::{llvm_objdump, run, rustc};

fn main() {
    rustc().input("eh_frame-terminator.rs").run();
    run("eh_frame-terminator").assert_stdout_contains("1122334455667788");
    llvm_objdump()
        .arg("--dwarf=frames")
        .input("eh_frame-terminator")
        .run()
        .assert_stdout_contains("ZERO terminator");
}
