use std::fs;
use std::path::{Path, PathBuf};

use crate::core::builder::{Builder, ShouldRun, Step};
use crate::t;

#[derive(Clone)]
pub struct RecordFailedTests {
    failed_tests_path: Option<PathBuf>,
}

impl RecordFailedTests {
    pub fn path(&self) -> Option<&Path> {
        self.failed_tests_path.as_deref()
    }
}

impl RecordFailedTests {}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct SetupFailedTestsFile;
impl Step for SetupFailedTestsFile {
    type Output = RecordFailedTests;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        if !builder.config.cmd.record() || builder.config.dry_run() {
            return RecordFailedTests { failed_tests_path: None };
        }

        let failed_tests_path = builder.out.join(&builder.config.record_failed_tests_path);
        println!(
            "setting up tracking of failed tests in {} (--record was passed)",
            failed_tests_path.display()
        );
        if failed_tests_path.exists() {
            println!("deleting previously recorded failed tests");
            t!(fs::remove_file(&failed_tests_path));
        }
        RecordFailedTests { failed_tests_path: Some(failed_tests_path) }
    }
}
