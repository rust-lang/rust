use std::path::PathBuf;

use super::{AllowUnused, Emit, LinkToAux, ProcRes, TestCx};

impl TestCx<'_> {
    pub(super) fn run_assembly_test(&self) {
        if self.config.llvm_filecheck.is_none() {
            self.fatal("missing --llvm-filecheck");
        }

        let (proc_res, output_path) = self.compile_test_and_save_assembly();
        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        let proc_res = self.verify_with_filecheck(&output_path);
        if !proc_res.status.success() {
            self.fatal_proc_rec("verification with 'FileCheck' failed", &proc_res);
        }
    }

    fn compile_test_and_save_assembly(&self) -> (ProcRes, PathBuf) {
        let output_file = self.get_output_file("s");
        let input_file = &self.testpaths.file;

        let mut emit = Emit::None;
        match self.props.assembly_output.as_ref().map(AsRef::as_ref) {
            Some("emit-asm") => {
                emit = Emit::Asm;
            }

            Some("bpf-linker") => {
                emit = Emit::LinkArgsAsm;
            }

            Some("ptx-linker") => {
                // No extra flags needed.
            }

            Some(header) => self.fatal(&format!("unknown 'assembly-output' header: {header}")),
            None => self.fatal("missing 'assembly-output' header"),
        }

        let rustc = self.make_compile_args(
            input_file,
            output_file,
            emit,
            AllowUnused::No,
            LinkToAux::Yes,
            Vec::new(),
        );

        let proc_res = self.compose_and_run_compiler(rustc, None, self.testpaths);
        let output_path = self.get_filecheck_file("s");
        (proc_res, output_path)
    }
}
