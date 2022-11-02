//! Tidy check to ensure that mir opt directories do not have stale files.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub fn check(path: &Path, bad: &mut bool) {
    let mut rs_files = Vec::<PathBuf>::new();
    let mut output_files = HashSet::<PathBuf>::new();
    let files = walkdir::WalkDir::new(&path.join("test/mir-opt")).into_iter();

    for file in files.filter_map(Result::ok).filter(|e| e.file_type().is_file()) {
        let filepath = file.path();
        if filepath.extension() == Some("rs".as_ref()) {
            rs_files.push(filepath.to_owned());
        } else {
            output_files.insert(filepath.to_owned());
        }
    }

    for file in rs_files {
        for bw in [32, 64] {
            for output_file in miropt_test_tools::files_for_miropt_test(&file, bw) {
                output_files.remove(&output_file.expected_file);
            }
        }
    }

    for extra in output_files {
        if extra.file_name() != Some("README.md".as_ref()) {
            tidy_error!(
                bad,
                "the following output file is not associated with any mir-opt test, you can remove it: {}",
                extra.display()
            );
        }
    }
}
