use std::process::Command;

use super::{TestCx, remove_and_create_dir_all};

impl TestCx<'_> {
    pub(super) fn run_rustdoc_json_test(&self) {
        //FIXME: Add bless option.

        assert!(self.revision.is_none(), "revisions not supported in this test suite");

        let out_dir = self.output_base_dir();
        remove_and_create_dir_all(&out_dir).unwrap_or_else(|e| {
            panic!("failed to remove and recreate output directory `{out_dir}`: {e}")
        });

        let proc_res = self.document(&out_dir, &self.testpaths);
        if !proc_res.status.success() {
            self.fatal_proc_rec("rustdoc failed!", &proc_res);
        }

        let mut json_out = out_dir.join(self.testpaths.file.file_stem().unwrap());
        json_out.set_extension("json");
        let res = self.run_command_to_procres(
            Command::new(self.config.jsondocck_path.as_ref().unwrap())
                .arg("--doc-dir")
                .arg(&out_dir)
                .arg("--template")
                .arg(&self.testpaths.file),
        );

        if !res.status.success() {
            self.fatal_proc_rec_general("jsondocck failed!", None, &res, || {
                writeln!(self.stdout, "Rustdoc Output:");
                writeln!(self.stdout, "{}", proc_res.format_info());
            })
        }

        let mut json_out = out_dir.join(self.testpaths.file.file_stem().unwrap());
        json_out.set_extension("json");

        let res = self.run_command_to_procres(
            Command::new(self.config.jsondoclint_path.as_ref().unwrap()).arg(&json_out),
        );

        if !res.status.success() {
            self.fatal_proc_rec("jsondoclint failed!", &res);
        }
    }
}
