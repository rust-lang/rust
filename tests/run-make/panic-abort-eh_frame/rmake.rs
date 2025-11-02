// An `.eh_frame` section in an object file is a symptom of an UnwindAction::Terminate
// being inserted, useful for determining whether or not unwinding is necessary.
// This is useless when panics would NEVER unwind due to -C panic=abort and when we don't need
// being able to generate backtraces (which depend on unwind tables on linux). This section should
// therefore never appear in the emit file of a -C panic=abort compilation
// with -C force-unwind-tables=no, and this test checks that this is respected.
// See https://github.com/rust-lang/rust/pull/112403 and
// https://github.com/rust-lang/rust/pull/143613.

//@ only-linux
// FIXME(Oneirical): the DW_CFA symbol appears on Windows-gnu, because uwtable
// is forced to true on Windows targets (see #128136).

use run_make_support::{llvm_objdump, rustc};

fn main() {
    rustc()
        .input("foo.rs")
        .crate_type("lib")
        .emit("obj=foo.o")
        .panic("abort")
        .edition("2021")
        .arg("-Zvalidate-mir")
        .arg("-Cforce-unwind-tables=no")
        .run();
    llvm_objdump().arg("--dwarf=frames").input("foo.o").run().assert_stdout_not_contains("DW_CFA");
}
