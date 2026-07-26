use super::{DocKind, TestCx, remove_and_create_dir_all};
use crate::util::ArgFileCommand;

impl TestCx<'_> {
    pub(super) fn run_rustdoc_json_test(&self) {
        //FIXME: Add bless option.

        assert!(self.variant.revision.is_none(), "revisions not supported in this test suite");

        let out_dir = self.output_base_dir();
        remove_and_create_dir_all(&out_dir).unwrap_or_else(|e| {
            panic!("failed to remove and recreate output directory `{out_dir}`: {e}")
        });

        let run_rustdoc = |doc_kind| {
            let proc_res = self.document(&out_dir, doc_kind);
            if !self.config.capture {
                writeln!(self.stdout, "{}", proc_res.format_info());
            }

            if !proc_res.status.success() {
                self.fatal_proc_rec("rustdoc failed!", &proc_res);
            }

            proc_res
        };

        let proc_res = run_rustdoc(DocKind::Json);

        let mut cmd = ArgFileCommand::new(self.config.jsondocck_path.as_ref().unwrap());
        cmd.arg("--doc-dir").arg(&out_dir).arg("--template").arg(&self.testpaths.file);
        let res = self.run_command_to_procres(cmd);

        if !res.status.success() {
            self.fatal_proc_rec_general("jsondocck failed!", None, &res, || {
                // FIXME: Include `--output-format=postcard` info here too.
                // Alternatively, ditch this section altogether, how useful is it?
                writeln!(self.stdout, "Rustdoc Output:");
                writeln!(self.stdout, "{}", proc_res.format_info());
            })
        }

        let mut json_out = out_dir.join(self.testpaths.file.file_stem().unwrap());
        json_out.set_extension("json");

        run_rustdoc(DocKind::Postcard);
        let mut postcard_out = json_out.clone();
        postcard_out.set_extension("postcard");

        let mut cmd = ArgFileCommand::new(self.config.jsondoclint_path.as_ref().unwrap());
        cmd.arg(&json_out);
        cmd.arg(&postcard_out);
        let res = self.run_command_to_procres(cmd);

        if !res.status.success() {
            self.fatal_proc_rec("jsondoclint failed!", &res);
        }
    }
}
