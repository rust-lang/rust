// Check if hotpatch only makes the functions hotpachable that were not,
// but leaving the other functions untouched
// More details in lib.rs

//@ revisions: x32 x64
//@[x32] only-x86
//@[x64] only-x86_64
// Reason: hotpatch is only implemented for X86

use run_make_support::{llvm, rustc};

fn main() {
    let disassemble_symbols_arg = "--disassemble-symbols=tailcall_fn,empty_fn";
    {
        rustc().input("lib.rs").crate_name("regular").crate_type("lib").opt_level("3").run();

        let regular_dump = llvm::llvm_objdump()
            .arg(disassemble_symbols_arg)
            .input("libregular.rlib")
            .run()
            .stdout_utf8();

        llvm::llvm_filecheck().patterns("lib.rs").stdin_buf(regular_dump).run();
    }

    {
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

        llvm::llvm_filecheck()
            .patterns("lib.rs")
            .check_prefix("HOTPATCH")
            .stdin_buf(hotpatch_dump)
            .run();
    }
}
