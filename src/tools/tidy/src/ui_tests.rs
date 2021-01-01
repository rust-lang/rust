//! Tidy check to ensure below in UI test directories:
//! - the number of entries in each directory must be less than `ENTRY_LIMIT`
//! - there are no stray `.stderr` files

use std::fs;
use std::path::Path;

const ENTRY_LIMIT: usize = 1000;
// FIXME: The following limits should be reduced eventually.
const ROOT_ENTRY_LIMIT: usize = 1500;
const ISSUES_ENTRY_LIMIT: usize = 2830;

fn check_entries(path: &Path, bad: &mut bool) {
    let dirs = walkdir::WalkDir::new(&path.join("test/ui"))
        .into_iter()
        .filter_entry(|e| e.file_type().is_dir());
    for dir in dirs {
        if let Ok(dir) = dir {
            let dir_path = dir.path();

            // Use special values for these dirs.
            let is_root = path.join("test/ui") == dir_path;
            let is_issues_dir = path.join("test/ui/issues") == dir_path;
            let limit = if is_root {
                ROOT_ENTRY_LIMIT
            } else if is_issues_dir {
                ISSUES_ENTRY_LIMIT
            } else {
                ENTRY_LIMIT
            };

            let count = std::fs::read_dir(dir_path).unwrap().count();
            if count >= limit {
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
}

pub fn check(path: &Path, bad: &mut bool) {
    check_entries(&path, bad);
    for path in &[&path.join("test/ui"), &path.join("test/ui-fulldeps")] {
        super::walk_no_read(path, &mut |_| false, &mut |entry| {
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
}
