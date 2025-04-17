use camino::Utf8PathBuf;

use super::{AllowUnused, Emit, LinkToAux, ProcRes, TargetLocation, TestCx};

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

    fn compile_test_and_save_assembly(&self) -> (ProcRes, Utf8PathBuf) {
        // This works with both `--emit asm` (as default output name for the assembly)
        // and `ptx-linker` because the latter can write output at requested location.
        let output_path = self.output_base_name().with_extension("s");
        let input_file = &self.testpaths.file;

        // Use the `//@ assembly-output:` directive to determine how to emit assembly.
        let emit = match self.props.assembly_output.as_deref() {
            Some("emit-asm") => Emit::Asm,
            Some("bpf-linker") => Emit::LinkArgsAsm,
            Some("ptx-linker") => Emit::None, // No extra flags needed.
            Some(other) => self.fatal(&format!("unknown 'assembly-output' directive: {other}")),
            None => self.fatal("missing 'assembly-output' directive"),
        };

        let rustc = self.make_compile_args(
            input_file,
            TargetLocation::ThisFile(output_path.clone()),
            emit,
            AllowUnused::No,
            LinkToAux::Yes,
            Vec::new(),
        );

        let proc_res = self.compose_and_run_compiler(rustc, None, self.testpaths);
        (proc_res, output_path)
    }
}
