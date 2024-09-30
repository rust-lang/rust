// Check if hotpatch only makes the functions hotpachable that were not,
// but leaving the other functions untouched
// More details in lib.rs

//@ revisions: x32 x64
//@[x32] only-x86
//@[x64] only-x86_64
// Reason: hotpatch is only implemented for X86

use run_make_support::{llvm, rustc};

fn main() {
    fn base_rustc() -> rustc::Rustc {
        let mut rustc = rustc();
        rustc.input("lib.rs").crate_type("lib").opt_level("3");
        rustc
    }

    fn dump_lib(libname: &str) -> String {
        llvm::llvm_objdump()
            .arg("--disassemble-symbols=tailcall_fn,empty_fn")
            .input(libname)
            .run()
            .stdout_utf8()
    }

    {
        base_rustc().crate_name("regular").run();
        let regular_dump = dump_lib("libregular.rlib");
        llvm::llvm_filecheck().patterns("lib.rs").stdin_buf(regular_dump).run();
    }

    {
        base_rustc().crate_name("hotpatch").arg("-Zhotpatch").run();
        let hotpatch_dump = dump_lib("libhotpatch.rlib");

        llvm::llvm_filecheck()
            .patterns("lib.rs")
            .check_prefix("HOTPATCH")
            .stdin_buf(hotpatch_dump)
            .run();
    }
}
