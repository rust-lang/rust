//! `bless` updates the reference files in the repo with changed output files
//! from the last test run.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::lazy::SyncLazy;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::clippy_project_root;

// NOTE: this is duplicated with tests/cargo/mod.rs What to do?
pub static CARGO_TARGET_DIR: SyncLazy<PathBuf> = SyncLazy::new(|| match env::var_os("CARGO_TARGET_DIR") {
    Some(v) => v.into(),
    None => env::current_dir().unwrap().join("target"),
});

pub fn bless() {
    let test_dirs = [
        clippy_project_root().join("tests").join("ui"),
        clippy_project_root().join("tests").join("ui-toml"),
        clippy_project_root().join("tests").join("ui-cargo"),
    ];
    for test_dir in &test_dirs {
        WalkDir::new(test_dir)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|f| f.path().extension() == Some(OsStr::new("rs")))
            .for_each(|f| {
                update_reference_file(f.path().with_extension("stdout"));
                update_reference_file(f.path().with_extension("stderr"));
                update_reference_file(f.path().with_extension("fixed"));
            });
    }
}

fn update_reference_file(reference_file_path: PathBuf) {
    let test_output_path = build_dir().join(PathBuf::from(reference_file_path.file_name().unwrap()));
    let relative_reference_file_path = reference_file_path.strip_prefix(clippy_project_root()).unwrap();

    // If compiletest did not write any changes during the test run,
    // we don't have to update anything
    if !test_output_path.exists() {
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

fn build_dir() -> PathBuf {
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let mut path = PathBuf::new();
    path.push(CARGO_TARGET_DIR.clone());
    path.push(profile);
    path.push("test_build_base");
    path
}
