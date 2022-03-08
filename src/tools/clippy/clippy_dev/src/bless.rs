//! `bless` updates the reference files in the repo with changed output files
//! from the last test run.

use std::ffi::OsStr;
use std::fs;
use std::lazy::SyncLazy;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

#[cfg(not(windows))]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy";
#[cfg(windows)]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy.exe";

static CLIPPY_BUILD_TIME: SyncLazy<Option<std::time::SystemTime>> = SyncLazy::new(|| {
    let mut path = std::env::current_exe().unwrap();
    path.set_file_name(CARGO_CLIPPY_EXE);
    fs::metadata(path).ok()?.modified().ok()
});

/// # Panics
///
/// Panics if the path to a test file is broken
pub fn bless(ignore_timestamp: bool) {
    let extensions = ["stdout", "stderr", "fixed"].map(OsStr::new);

    WalkDir::new(build_dir())
        .into_iter()
        .map(Result::unwrap)
        .filter(|entry| entry.path().extension().map_or(false, |ext| extensions.contains(&ext)))
        .for_each(|entry| update_reference_file(&entry, ignore_timestamp));
}

fn update_reference_file(test_output_entry: &DirEntry, ignore_timestamp: bool) {
    let test_output_path = test_output_entry.path();

    let reference_file_name = test_output_entry.file_name().to_str().unwrap().replace(".stage-id", "");
    let reference_file_path = Path::new("tests")
        .join(test_output_path.strip_prefix(build_dir()).unwrap())
        .with_file_name(reference_file_name);

    // If the test output was not updated since the last clippy build, it may be outdated
    if !ignore_timestamp && !updated_since_clippy_build(test_output_entry).unwrap_or(true) {
        return;
    }

    let test_output_file = fs::read(&test_output_path).expect("Unable to read test output file");
    let reference_file = fs::read(&reference_file_path).unwrap_or_default();

    if test_output_file != reference_file {
        // If a test run caused an output file to change, update the reference file
        println!("updating {}", reference_file_path.display());
        fs::copy(test_output_path, &reference_file_path).expect("Could not update reference file");
    }
}

fn updated_since_clippy_build(entry: &DirEntry) -> Option<bool> {
    let clippy_build_time = (*CLIPPY_BUILD_TIME)?;
    let modified = entry.metadata().ok()?.modified().ok()?;
    Some(modified >= clippy_build_time)
}

fn build_dir() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.set_file_name("test");
    path
}
