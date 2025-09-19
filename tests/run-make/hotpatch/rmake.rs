// Check if hotpatch makes the functions hotpachable that were not
// More details in lib.rs

use run_make_support::{llvm, rustc};

fn main() {
    // hotpatch is only implemented for x86 and aarch64, but for aarch64 functions
    // are always hotpatchable so we don't need to check it
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        fn base_rustc() -> rustc::Rustc {
            let mut rustc = rustc();
            rustc.input("lib.rs").crate_type("lib").opt_level("3");
            rustc
        }

        fn dump_lib(libname: &str) -> String {
            llvm::llvm_objdump()
                .arg("--disassemble-symbols=empty_fn")
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
}
