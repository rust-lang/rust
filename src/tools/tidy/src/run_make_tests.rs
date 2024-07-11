//! Tidy check to ensure that no new Makefiles are added under `tests/run-make/`.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn check(tests_path: &Path, src_path: &Path, bless: bool, bad: &mut bool) {
    let mut is_sorted = true;

    let allowed_makefiles = {
        let mut total_lines = 0;
        let mut prev_line = "";
        let allowed_makefiles: BTreeSet<&str> = include_str!("allowed_run_make_makefiles.txt")
            .lines()
            .map(|line| {
                total_lines += 1;

                if prev_line > line {
                    is_sorted = false;
                }

                prev_line = line;

                line
            })
            .collect();

        if !is_sorted && !bless {
            tidy_error!(
                bad,
                "`src/tools/tidy/src/allowed_run_make_makefiles.txt` is not in order, likely \
                because you modified it manually, please only update it with command \
                `x test tidy --bless`"
            );
        }
        if allowed_makefiles.len() != total_lines {
            tidy_error!(
                bad,
                "`src/tools/tidy/src/allowed_run_make_makefiles.txt` contains duplicate entries, \
                likely because you modified it manually, please only update it with command \
                `x test tidy --bless`"
            );
        }

        allowed_makefiles
    };

    let mut remaining_makefiles = allowed_makefiles.clone();

    crate::walk::walk_no_read(
        &[tests_path.join("run-make").as_ref()],
        |_, _| false,
        &mut |entry| {
            if entry.file_type().map_or(true, |t| t.is_dir()) {
                return;
            }

            if entry.file_name().to_str().map_or(true, |f| f != "Makefile") {
                return;
            }

            let makefile_path = entry.path().strip_prefix(&tests_path).unwrap();
            let makefile_path = makefile_path.to_str().unwrap().replace('\\', "/");

            if !remaining_makefiles.remove(makefile_path.as_str()) {
                tidy_error!(
                    bad,
                    "found run-make Makefile not permitted in \
                `src/tools/tidy/src/allowed_run_make_makefiles.txt`, please write new run-make \
                tests with `rmake.rs` instead: {}",
                    entry.path().display()
                );
            }
        },
    );

    // If there are any expected Makefiles remaining, they were moved or deleted.
    // Our data must remain up to date, so they must be removed from
    // `src/tools/tidy/src/allowed_run_make_makefiles.txt`.
    // This can be done automatically on --bless, or else a tidy error will be issued.
    if bless && (!remaining_makefiles.is_empty() || !is_sorted) {
        let tidy_src = src_path.join("tools").join("tidy").join("src");
        let org_file_path = tidy_src.join("allowed_run_make_makefiles.txt");
        let temp_file_path = tidy_src.join("blessed_allowed_run_make_makefiles.txt");
        let mut temp_file = t!(File::create_new(&temp_file_path));
        for file in allowed_makefiles.difference(&remaining_makefiles) {
            t!(writeln!(temp_file, "{file}"));
        }
        t!(std::fs::rename(&temp_file_path, &org_file_path));
    } else {
        for file in remaining_makefiles {
            let mut p = PathBuf::from(tests_path);
            p.push(file);
            tidy_error!(
                bad,
                "Makefile `{}` no longer exists and should be removed from the exclusions in \
                `src/tools/tidy/src/allowed_run_make_makefiles.txt`, you can run `x test tidy --bless` to update \
                the allow list",
                p.display()
            );
        }
    }
}
