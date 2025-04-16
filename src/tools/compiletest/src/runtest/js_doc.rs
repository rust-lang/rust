use std::process::Command;

use super::TestCx;

impl TestCx<'_> {
    pub(super) fn run_rustdoc_js_test(&self) {
        if let Some(nodejs) = &self.config.nodejs {
            let out_dir = self.output_base_dir();

            self.document(&out_dir, &self.testpaths);

            let file_stem = self.testpaths.file.file_stem().expect("no file stem");
            let res = self.run_command_to_procres(
                Command::new(&nodejs)
                    .arg(self.config.src_root.join("src/tools/rustdoc-js/tester.js"))
                    .arg("--doc-folder")
                    .arg(out_dir)
                    .arg("--crate-name")
                    .arg(file_stem.replace("-", "_"))
                    .arg("--test-file")
                    .arg(self.testpaths.file.with_extension("js")),
            );
            if !res.status.success() {
                self.fatal_proc_rec("rustdoc-js test failed!", &res);
            }
        } else {
            self.fatal("no nodeJS");
        }
    }
}
