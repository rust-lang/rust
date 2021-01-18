//! `bless` updates the reference files in the repo with changed output files
//! from the last test run.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::lazy::SyncLazy;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::clippy_project_root;

// NOTE: this is duplicated with tests/cargo/mod.rs What to do?
pub static CARGO_TARGET_DIR: SyncLazy<PathBuf> = SyncLazy::new(|| match env::var_os("CARGO_TARGET_DIR") {
    Some(v) => v.into(),
    None => env::current_dir().unwrap().join("target"),
});

static CLIPPY_BUILD_TIME: SyncLazy<Option<std::time::SystemTime>> = SyncLazy::new(|| {
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let mut path = PathBuf::from(&**CARGO_TARGET_DIR);
    path.push(profile);
    path.push("cargo-clippy");
    fs::metadata(path).ok()?.modified().ok()
});

pub fn bless(ignore_timestamp: bool) {
    let test_suite_dirs = [
        clippy_project_root().join("tests").join("ui"),
        clippy_project_root().join("tests").join("ui-internal"),
        clippy_project_root().join("tests").join("ui-toml"),
        clippy_project_root().join("tests").join("ui-cargo"),
    ];
    for test_suite_dir in &test_suite_dirs {
        WalkDir::new(test_suite_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|f| f.path().extension() == Some(OsStr::new("rs")))
            .for_each(|f| {
                let test_name = f.path().strip_prefix(test_suite_dir).unwrap();
                for &ext in &["stdout", "stderr", "fixed"] {
                    update_reference_file(
                        f.path().with_extension(ext),
                        test_name.with_extension(ext),
                        ignore_timestamp,
                    );
                }
            });
    }
}

fn update_reference_file(reference_file_path: PathBuf, test_name: PathBuf, ignore_timestamp: bool) {
    let test_output_path = build_dir().join(test_name);
    let relative_reference_file_path = reference_file_path.strip_prefix(clippy_project_root()).unwrap();

    // If compiletest did not write any changes during the test run,
    // we don't have to update anything
    if !test_output_path.exists() {
        return;
    }

    // If the test output was not updated since the last clippy build, it may be outdated
    if !ignore_timestamp && !updated_since_clippy_build(&test_output_path).unwrap_or(true) {
        return;
    }

    let test_output_file = fs::read(&test_output_path).expect("Unable to read test output file");
    let reference_file = fs::read(&reference_file_path).unwrap_or_default();

    if test_output_file != reference_file {
        // If a test run caused an output file to change, update the reference file
        println!("updating {}", &relative_reference_file_path.display());
        fs::copy(test_output_path, &reference_file_path).expect("Could not update reference file");

        // We need to re-read the file now because it was potentially updated from copying
        let reference_file = fs::read(&reference_file_path).unwrap_or_default();

        if reference_file.is_empty() {
            // If we copied over an empty output file, we remove the now empty reference file
            println!("removing {}", &relative_reference_file_path.display());
            fs::remove_file(reference_file_path).expect("Could not remove reference file");
        }
    }
}

fn updated_since_clippy_build(path: &Path) -> Option<bool> {
    let clippy_build_time = (*CLIPPY_BUILD_TIME)?;
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    Some(modified >= clippy_build_time)
}

fn build_dir() -> PathBuf {
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let mut path = PathBuf::new();
    path.push(CARGO_TARGET_DIR.clone());
    path.push(profile);
    path.push("test_build_base");
    path
}
