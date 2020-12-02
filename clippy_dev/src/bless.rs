//! `bless` updates the 'expected output' files in the repo with changed output files
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
                update_test_file(f.path().with_extension("stdout"));
                update_test_file(f.path().with_extension("stderr"));
                update_test_file(f.path().with_extension("fixed"));
            });
    }
}

fn update_test_file(test_file_path: PathBuf) {
    let build_output_path = build_dir().join(PathBuf::from(test_file_path.file_name().unwrap()));
    let relative_test_file_path = test_file_path.strip_prefix(clippy_project_root()).unwrap();

    // If compiletest did not write any changes during the test run,
    // we don't have to update anything
    if !build_output_path.exists() {
        return;
    }

    let build_output = fs::read(&build_output_path).expect("Unable to read build output file");
    let test_file = fs::read(&test_file_path).expect("Unable to read test file");

    if build_output != test_file {
        // If a test run caused an output file to change, update the test file
        println!("updating {}", &relative_test_file_path.display());
        fs::copy(build_output_path, &test_file_path).expect("Could not update test file");

        if test_file.is_empty() {
            // If we copied over an empty output file, we remove it
            println!("removing {}", &relative_test_file_path.display());
            fs::remove_file(test_file_path).expect("Could not remove test file");
        }
    }
}

fn build_dir() -> PathBuf {
    let profile = format!("{}", env::var("PROFILE").unwrap_or("debug".to_string()));
    let mut path = PathBuf::new();
    path.push(CARGO_TARGET_DIR.clone());
    path.push(profile);
    path.push("test_build_base");
    path
}
