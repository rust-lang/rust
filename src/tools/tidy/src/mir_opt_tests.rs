//! Tidy check to ensure that mir opt directories do not have stale files or dashes in file names

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use miropt_test_tools::PanicStrategy;

use crate::walk::walk_no_read;

fn check_unused_files(path: &Path, bless: bool, bad: &mut bool) {
    let mut rs_files = Vec::<PathBuf>::new();
    let mut output_files = HashSet::<PathBuf>::new();

    walk_no_read(
        &[&path.join("mir-opt")],
        |path, _is_dir| path.file_name() == Some("README.md".as_ref()),
        &mut |file| {
            let filepath = file.path();
            if filepath.extension() == Some("rs".as_ref()) {
                rs_files.push(filepath.to_owned());
            } else {
                output_files.insert(filepath.to_owned());
            }
        },
    );

    for file in rs_files {
        for bw in [32, 64] {
            for ps in [PanicStrategy::Unwind, PanicStrategy::Abort] {
                let mir_opt_test = miropt_test_tools::files_for_miropt_test(&file, bw, ps);
                for output_file in mir_opt_test.files {
                    output_files.remove(&output_file.expected_file);
                }
            }
        }
    }

    for extra in output_files {
        if !bless {
            tidy_error!(
                bad,
                "the following output file is not associated with any mir-opt test, you can remove it: {}",
                extra.display()
            );
        } else {
            let _ = std::fs::remove_file(extra);
        }
    }
}

fn check_dash_files(path: &Path, bless: bool, bad: &mut bool) {
    for file in walkdir::WalkDir::new(path.join("mir-opt"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let path = file.path();
        if path.extension() == Some("rs".as_ref())
            && let Some(name) = path.file_name().and_then(|s| s.to_str())
            && name.contains('-')
        {
            if !bless {
                tidy_error!(
                    bad,
                    "mir-opt test files should not have dashes in them: {}",
                    path.display()
                );
            } else {
                let new_name = name.replace('-', "_");
                let mut new_path = path.to_owned();
                new_path.set_file_name(new_name);
                let _ = std::fs::rename(path, new_path);
            }
        }
    }
}

pub fn check(path: &Path, bless: bool, bad: &mut bool) {
    check_unused_files(path, bless, bad);
    check_dash_files(path, bless, bad);
}
