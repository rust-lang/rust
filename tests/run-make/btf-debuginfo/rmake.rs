//! The `.BTF` section is printed with `print_btf.py`, copied verbatim from LLVM's
//! `llvm/test/CodeGen/BPF/BTF/print_btf.py`, so that the expectations in test files
//! read like LLVM's own BPF BTF tests.

//@ needs-llvm-components: bpf
//@ ignore-backends: gcc

use run_make_support::{
    cmd, llvm, llvm_filecheck, llvm_nm, llvm_objcopy, llvm_objdump, python_command, rustc,
    rustc_minicore,
};

fn main() {
    for target in ["bpfel-unknown-none", "bpfeb-unknown-none"] {
        rustc_minicore().target(target).output("libminicore.rlib").run();

        for test in ["extern", "extern_weak"] {
            let src = format!("{test}.rs");
            let bc = format!("{test}.bc");
            let obj = format!("{test}.o");
            let btf = format!("{test}.btf");

            // For BPF targets, rustc's "object" files are LLVM bitcode meant for bpf-linker, so use
            // `llc` to get the ELF object containing the `.BTF` section.
            rustc()
                .input(&src)
                .target(target)
                .emit("obj")
                .debuginfo("2")
                .opt_level("2")
                .extern_("minicore", "libminicore.rlib")
                .output(&bc)
                .run();
            cmd(llvm::llvm_bin_dir().join("llc")).args(["-filetype=obj", &bc, "-o", &obj]).run();
            llvm_objcopy().dump_section(".BTF", &btf).arg(&obj).run();

            let dump = python_command().arg("print_btf.py").arg(&btf).run().stdout_utf8();
            llvm_filecheck().patterns(&src).check_prefix("CHECK-BTF").stdin_buf(dump).run();

            if test == "extern_weak" {
                let disasm = llvm_objdump()
                    .disassemble()
                    .args(["-r", "--no-show-raw-insn"])
                    .input(&obj)
                    .run()
                    .stdout_utf8();
                llvm_filecheck().patterns(&src).check_prefix("CHECK-OBJ").stdin_buf(disasm).run();

                let syms = llvm_nm().input(&obj).run().stdout_utf8();
                llvm_filecheck().patterns(&src).check_prefix("CHECK-SYM").stdin_buf(syms).run();
            }
        }
    }
}
