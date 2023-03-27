//! Tidy check to ensure below in UI test directories:
//! - the number of entries in each directory must be less than `ENTRY_LIMIT`
//! - there are no stray `.stderr` files

use ignore::Walk;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const ENTRY_LIMIT: usize = 1000;
// FIXME: The following limits should be reduced eventually.
const ROOT_ENTRY_LIMIT: usize = 940;
const ISSUES_ENTRY_LIMIT: usize = 1978;

fn check_entries(tests_path: &Path, bad: &mut bool) {
    let mut directories: HashMap<PathBuf, usize> = HashMap::new();

    for dir in Walk::new(&tests_path.join("ui")) {
        if let Ok(entry) = dir {
            let parent = entry.path().parent().unwrap().to_path_buf();
            *directories.entry(parent).or_default() += 1;
        }
    }

    for (dir_path, count) in directories {
        // Use special values for these dirs.
        let is_root = tests_path.join("ui") == dir_path;
        let is_issues_dir = tests_path.join("ui/issues") == dir_path;
        let limit = if is_root {
            ROOT_ENTRY_LIMIT
        } else if is_issues_dir {
            ISSUES_ENTRY_LIMIT
        } else {
            ENTRY_LIMIT
        };

        if count > limit {
            tidy_error!(
                bad,
                "following path contains more than {} entries, \
                    you should move the test to some relevant subdirectory (current: {}): {}",
                limit,
                count,
                dir_path.display()
            );
        }
    }
}

pub fn check(path: &Path, bad: &mut bool) {
    check_entries(&path, bad);
    let (ui, ui_fulldeps) = (path.join("ui"), path.join("ui-fulldeps"));
    let paths = [ui.as_path(), ui_fulldeps.as_path()];
    crate::walk::walk_no_read(&paths, |_, _| false, &mut |entry| {
        let file_path = entry.path();
        if let Some(ext) = file_path.extension() {
            if ext == "stderr" || ext == "stdout" {
                // Test output filenames have one of the formats:
                // ```
                // $testname.stderr
                // $testname.$mode.stderr
                // $testname.$revision.stderr
                // $testname.$revision.$mode.stderr
                // ```
                //
                // For now, just make sure that there is a corresponding
                // `$testname.rs` file.
                //
                // NB: We do not use file_stem() as some file names have multiple `.`s and we
                // must strip all of them.
                let testname =
                    file_path.file_name().unwrap().to_str().unwrap().split_once('.').unwrap().0;
                if !file_path.with_file_name(testname).with_extension("rs").exists() {
                    tidy_error!(bad, "Stray file with UI testing output: {:?}", file_path);
                }

                if let Ok(metadata) = fs::metadata(file_path) {
                    if metadata.len() == 0 {
                        tidy_error!(bad, "Empty file with UI testing output: {:?}", file_path);
                    }
                }
            }
        }
    });
}
