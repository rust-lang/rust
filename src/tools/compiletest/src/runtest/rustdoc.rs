use std::process::Command;

use super::{TestCx, remove_and_create_dir_all};

impl TestCx<'_> {
    pub(super) fn run_rustdoc_test(&self) {
        assert!(self.revision.is_none(), "revisions not relevant here");

        let out_dir = self.output_base_dir();
        remove_and_create_dir_all(&out_dir);

        let proc_res = self.document(&out_dir, &self.testpaths);
        if !proc_res.status.success() {
            self.fatal_proc_rec("rustdoc failed!", &proc_res);
        }

        if self.props.check_test_line_numbers_match {
            self.check_rustdoc_test_option(proc_res);
        } else {
            let root = self.config.find_rust_src_root().unwrap();
            let mut cmd = Command::new(&self.config.python);
            cmd.arg(root.join("src/etc/htmldocck.py")).arg(&out_dir).arg(&self.testpaths.file);
            if self.config.bless {
                cmd.arg("--bless");
            }
            let res = self.run_command_to_procres(&mut cmd);
            if !res.status.success() {
                self.fatal_proc_rec_with_ctx("htmldocck failed!", &res, |mut this| {
                    this.compare_to_default_rustdoc(&out_dir)
                });
            }
        }
    }
}
