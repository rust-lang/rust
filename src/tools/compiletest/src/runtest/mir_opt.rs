use std::fs;
use std::path::{Path, PathBuf};

use glob::glob;
use miropt_test_tools::{MiroptTest, MiroptTestFile, files_for_miropt_test};
use tracing::debug;

use super::{Emit, TestCx, WillExecute};
use crate::compute_diff::write_diff;

impl TestCx<'_> {
    pub(super) fn run_mir_opt_test(&self) {
        let pm = self.pass_mode();
        let should_run = self.should_run(pm);

        let mut test_info = files_for_miropt_test(
            &self.testpaths.file,
            self.config.get_pointer_width(),
            self.config.target_cfg().panic.for_miropt_test_tools(),
        );

        let passes = std::mem::take(&mut test_info.passes);

        let proc_res = self.compile_test_with_passes(should_run, Emit::Mir, passes);
        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }
        self.check_mir_dump(test_info);

        if let WillExecute::Yes = should_run {
            let proc_res = self.exec_compiled_test();

            if !proc_res.status.success() {
                self.fatal_proc_rec("test run failed!", &proc_res);
            }
        }
    }

    fn check_mir_dump(&self, test_info: MiroptTest) {
        let test_dir = self.testpaths.file.parent().unwrap();
        let test_crate =
            self.testpaths.file.file_stem().unwrap().to_str().unwrap().replace('-', "_");

        let MiroptTest { run_filecheck, suffix, files, passes: _ } = test_info;

        if self.config.bless {
            for e in
                glob(&format!("{}/{}.*{}.mir", test_dir.display(), test_crate, suffix)).unwrap()
            {
                fs::remove_file(e.unwrap()).unwrap();
            }
            for e in
                glob(&format!("{}/{}.*{}.diff", test_dir.display(), test_crate, suffix)).unwrap()
            {
                fs::remove_file(e.unwrap()).unwrap();
            }
        }

        for MiroptTestFile { from_file, to_file, expected_file } in files {
            let dumped_string = if let Some(after) = to_file {
                self.diff_mir_files(from_file.into(), after.into())
            } else {
                let mut output_file = PathBuf::new();
                output_file.push(self.get_mir_dump_dir());
                output_file.push(&from_file);
                debug!(
                    "comparing the contents of: {} with {}",
                    output_file.display(),
                    expected_file.display()
                );
                if !output_file.exists() {
                    panic!(
                        "Output file `{}` from test does not exist, available files are in `{}`",
                        output_file.display(),
                        output_file.parent().unwrap().display()
                    );
                }
                self.check_mir_test_timestamp(&from_file, &output_file);
                let dumped_string = fs::read_to_string(&output_file).unwrap();
                self.normalize_output(&dumped_string, &[])
            };

            if self.config.bless {
                let _ = fs::remove_file(&expected_file);
                fs::write(expected_file, dumped_string.as_bytes()).unwrap();
            } else {
                if !expected_file.exists() {
                    panic!("Output file `{}` from test does not exist", expected_file.display());
                }
                let expected_string = fs::read_to_string(&expected_file).unwrap();
                if dumped_string != expected_string {
                    print!("{}", write_diff(&expected_string, &dumped_string, 3));
                    panic!(
                        "Actual MIR output differs from expected MIR output {}",
                        expected_file.display()
                    );
                }
            }
        }

        if run_filecheck {
            let output_path = self.output_base_name().with_extension("mir");
            let proc_res = self.verify_with_filecheck(&output_path);
            if !proc_res.status.success() {
                self.fatal_proc_rec("verification with 'FileCheck' failed", &proc_res);
            }
        }
    }

    fn diff_mir_files(&self, before: PathBuf, after: PathBuf) -> String {
        let to_full_path = |path: PathBuf| {
            let full = self.get_mir_dump_dir().join(&path);
            if !full.exists() {
                panic!(
                    "the mir dump file for {} does not exist (requested in {})",
                    path.display(),
                    self.testpaths.file.display(),
                );
            }
            full
        };
        let before = to_full_path(before);
        let after = to_full_path(after);
        debug!("comparing the contents of: {} with {}", before.display(), after.display());
        let before = fs::read_to_string(before).unwrap();
        let after = fs::read_to_string(after).unwrap();
        let before = self.normalize_output(&before, &[]);
        let after = self.normalize_output(&after, &[]);
        let mut dumped_string = String::new();
        for result in diff::lines(&before, &after) {
            use std::fmt::Write;
            match result {
                diff::Result::Left(s) => writeln!(dumped_string, "- {}", s).unwrap(),
                diff::Result::Right(s) => writeln!(dumped_string, "+ {}", s).unwrap(),
                diff::Result::Both(s, _) => writeln!(dumped_string, "  {}", s).unwrap(),
            }
        }
        dumped_string
    }

    fn check_mir_test_timestamp(&self, test_name: &str, output_file: &Path) {
        let t = |file| fs::metadata(file).unwrap().modified().unwrap();
        let source_file = &self.testpaths.file;
        let output_time = t(output_file);
        let source_time = t(source_file);
        if source_time > output_time {
            debug!("source file time: {:?} output file time: {:?}", source_time, output_time);
            panic!(
                "test source file `{}` is newer than potentially stale output file `{}`.",
                source_file.display(),
                test_name
            );
        }
    }
}
