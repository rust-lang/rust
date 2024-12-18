use std::process::Command;

use super::{TestCx, remove_and_create_dir_all};

impl TestCx<'_> {
    pub(super) fn run_rustdoc_json_test(&self) {
        //FIXME: Add bless option.

        assert!(self.revision.is_none(), "revisions not relevant here");

        let out_dir = self.output_base_dir();
        remove_and_create_dir_all(&out_dir);

        let proc_res = self.document(&out_dir, &self.testpaths);
        if !proc_res.status.success() {
            self.fatal_proc_rec("rustdoc failed!", &proc_res);
        }

        let root = self.config.find_rust_src_root().unwrap();
        let mut json_out = out_dir.join(self.testpaths.file.file_stem().unwrap());
        json_out.set_extension("json");
        let res = self.run_command_to_procres(
            Command::new(self.config.jsondocck_path.as_ref().unwrap())
                .arg("--doc-dir")
                .arg(root.join(&out_dir))
                .arg("--template")
                .arg(&self.testpaths.file),
        );

        if !res.status.success() {
            self.fatal_proc_rec_with_ctx("jsondocck failed!", &res, |_| {
                println!("Rustdoc Output:");
                proc_res.print_info();
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
