use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::{Path, PathBuf};

use crate::core::builder::{Builder, ShouldRun, Step};
use crate::t;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum IsForRerunningTests {
    /// For steps that aren't part of tests, same as no
    DontCare,
    No,
    Yes,
}

#[derive(Clone)]
pub struct RecordFailedTests {
    failed_tests_path: Option<PathBuf>,
}

impl RecordFailedTests {
    pub fn path(&self) -> Option<&Path> {
        self.failed_tests_path.as_deref()
    }
}

/// This step is run as a dependency of most testing steps.
/// Upon running, a file is created for failed tests to be recorded in if `--record` is passed on
/// the command line.
///
/// This step is the only way to get access to a token type called [`RecordFailedTests`].
/// Having this token type signifies the fact that a file was created to store failed tests in,
/// and is required to create a `Renderer`, the type that renders the outputs of tests.
///
/// If `--rerun` isn't passed, or we're in dry-run mode, running this step is a no-op,
/// and the `RecordFailedTest` type doesn't (need to) signify anything.
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

        let failed_tests_path = builder.config.record_failed_tests_path.clone();
        println!(
            "setting up tracking of failed tests in {} (`--record` was passed)",
            failed_tests_path.display()
        );
        if failed_tests_path.exists() {
            println!("deleting previously recorded failed tests");
            t!(fs::remove_file(&failed_tests_path));
        }
        RecordFailedTests { failed_tests_path: Some(failed_tests_path) }
    }
}

pub fn collect_previously_failed_tests(failed_tests_file_path: &PathBuf) -> Vec<PathBuf> {
    let mut paths = Vec::new();

    println!(
        "`--rerun` passed so looking for failed tests in {}",
        failed_tests_file_path.display()
    );

    let lines: Vec<String> = match File::open(failed_tests_file_path) {
        Ok(f) => t!(BufReader::new(f).lines().collect()),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            println!(
                "WARNING: failed tests file doesn't exist: `--rerun` only makes sense after a previous test run with `--record`"
            );
            return Vec::new();
        }
        Err(e) => t!(Err(e)),
    };

    let mut set_tracking_duplicates = HashSet::new();
    let mut num_printed = 0;
    const MAX_RERUN_PRINTS: usize = 10;

    for line in lines {
        let trimmed = line.as_str().trim();
        let without_revision =
            trimmed.rsplit_once("#").map(|(before, _)| before).unwrap_or(trimmed);
        let without_suite_prefix = without_revision
            .strip_prefix("[")
            .and_then(|rest| rest.split_once("]"))
            .map(|(_, after)| after.trim())
            .unwrap_or(without_revision);

        let failed_test_path = PathBuf::from(without_suite_prefix.to_string());
        if set_tracking_duplicates.insert(failed_test_path.clone()) {
            if num_printed == 0 {
                println!("rerunning previously failed tests:");
            }
            if num_printed < MAX_RERUN_PRINTS {
                println!("    {}", failed_test_path.display());
                num_printed += 1;
            }
            paths.push(failed_test_path);
        }
    }

    if num_printed == MAX_RERUN_PRINTS && set_tracking_duplicates.len() > MAX_RERUN_PRINTS {
        println!("    and {} more...", set_tracking_duplicates.len() - MAX_RERUN_PRINTS)
    }

    if set_tracking_duplicates.is_empty() {
        println!(
            "WARNING: failed tests file doesn't contain any failed tests: `--rerun` only makes sense after a previous test run with `--record`"
        );
    }

    paths
}
