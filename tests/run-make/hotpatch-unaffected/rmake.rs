// Check if hotpatch leaves the functions that are already hotpatchable untouched

//@ revisions: x32 x64
//@[x32] only-x86
//@[x64] only-x86_64
// Reason: hotpatch is only implemented for X86

use run_make_support::{assertion_helpers, llvm, rustc};

fn main() {
    let disassemble_symbols_arg = "--disassemble-symbols=return_42,tailcall";

    rustc().input("lib.rs").crate_name("regular").crate_type("lib").opt_level("3").run();

    let regular_dump = llvm::llvm_objdump()
        .arg(disassemble_symbols_arg)
        .input("libregular.rlib")
        .run()
        .stdout_utf8();

    rustc()
        .input("lib.rs")
        .crate_name("hotpatch")
        .crate_type("lib")
        .opt_level("3")
        .arg("-Zhotpatch")
        .run();

    let hotpatch_dump = llvm::llvm_objdump()
        .arg(disassemble_symbols_arg)
        .input("libhotpatch.rlib")
        .run()
        .stdout_utf8();

    {
        let mut lines_regular = regular_dump.lines();
        let mut lines_hotpatch = hotpatch_dump.lines();

        loop {
            match (lines_regular.next(), lines_hotpatch.next()) {
                (None, None) => break,
                (Some(r), Some(h)) => {
                    if r.contains("libregular.rlib") {
                        assertion_helpers::assert_contains(h, "libhotpatch.rlib")
                    } else {
                        assertion_helpers::assert_equals(&r, &h)
                    }
                }
                _ => panic!("the files should have equal length"),
            }
        }
    }
}
