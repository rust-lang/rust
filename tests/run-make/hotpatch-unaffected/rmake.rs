// Check if hotpatch leaves the functions that are already hotpatchable untouched

use run_make_support::{assertion_helpers, llvm, rustc};

fn main() {
    // hotpatch is only implemented for X86 and aarch64
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        fn base_rustc() -> rustc::Rustc {
            let mut rustc = rustc();
            rustc.input("lib.rs").crate_type("lib").opt_level("3");
            rustc
        }

        fn dump_lib(libname: &str) -> String {
            llvm::llvm_objdump()
                .arg("--disassemble-symbols=return_42,tailcall")
                .input(libname)
                .run()
                .stdout_utf8()
        }

        base_rustc().crate_name("regular").run();
        let regular_dump = dump_lib("libregular.rlib");

        base_rustc().crate_name("hotpatch").arg("-Zhotpatch").run();
        let hotpatch_dump = dump_lib("libhotpatch.rlib");

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
                    _ => panic!("expected files to have equal number of lines"),
                }
            }
        }
    }
}
