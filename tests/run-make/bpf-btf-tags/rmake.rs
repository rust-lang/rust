//! Checks the BTF produced for `#[btf_decl_tag]` and `#[btf_type_tag]` when compiling for BPF.

//@ needs-llvm-components: bpf
//@ ignore-backends: gcc

use run_make_support::{
    cmd, llvm, llvm_filecheck, llvm_objcopy, python_command, rustc, rustc_minicore,
};

fn main() {
    for target in ["bpfel-unknown-none", "bpfeb-unknown-none"] {
        rustc_minicore().target(target).output("libminicore.rlib").run();

        // For BPF targets, rustc's "object" files are LLVM bitcode meant for bpf-linker, so use
        // `llc` to get the ELF object containing the `.BTF` section.
        rustc()
            .input("lib.rs")
            .target(target)
            .emit("obj")
            .debuginfo("2")
            .opt_level("0")
            .extern_("minicore", "libminicore.rlib")
            .output("lib.bc")
            .run();
        cmd(llvm::llvm_bin_dir().join("llc"))
            .args(["-filetype=obj", "lib.bc", "-o", "lib.o"])
            .run();
        llvm_objcopy().dump_section(".BTF", "lib.btf").arg("lib.o").run();

        let btf = python_command().arg("print_btf.py").arg("lib.btf").run().stdout_utf8();
        llvm_filecheck().patterns("lib.rs").check_prefix("CHECK-BTF").stdin_buf(btf).run();
    }
}
