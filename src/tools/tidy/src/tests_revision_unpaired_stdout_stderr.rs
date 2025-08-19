//! Checks that there are no unpaired `.stderr` or `.stdout` for a test with and without revisions.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::path::Path;

use crate::iter_header::*;
use crate::walk::*;

// Should be kept in sync with `CompareMode` in `src/tools/compiletest/src/common.rs`,
// as well as `run`.
const IGNORES: &[&str] = &[
    "polonius",
    "chalk",
    "split-dwarf",
    "split-dwarf-single",
    "next-solver-coherence",
    "next-solver",
    "run",
];
const EXTENSIONS: &[&str] = &["stdout", "stderr"];
const SPECIAL_TEST: &str = "tests/ui/command/need-crate-arg-ignore-tidy.x.rs";

pub fn check(tests_path: impl AsRef<Path>, bad: &mut bool) {
    // Recurse over subdirectories under `tests/`
    walk_dir(tests_path.as_ref(), filter, &mut |entry| {
        // We are inspecting a folder. Collect the paths to interesting files `.rs`, `.stderr`,
        // `.stdout` under the current folder (shallow).
        let mut files_under_inspection = BTreeSet::new();
        for sibling in std::fs::read_dir(entry.path()).unwrap() {
            let Ok(sibling) = sibling else {
                continue;
            };

            if sibling.path().is_dir() {
                continue;
            }

            let sibling_path = sibling.path();

            let Some(ext) = sibling_path.extension().and_then(OsStr::to_str) else {
                continue;
            };

            if ext == "rs" || EXTENSIONS.contains(&ext) {
                files_under_inspection.insert(sibling_path);
            }
        }

        let mut test_info = BTreeMap::new();

        for test in
            files_under_inspection.iter().filter(|f| f.extension().is_some_and(|ext| ext == "rs"))
        {
            if test.ends_with(SPECIAL_TEST) {
                continue;
            }

            let mut expected_revisions = BTreeSet::new();

            let Ok(contents) = std::fs::read_to_string(test) else { continue };

            // Collect directives.
            iter_header(&contents, &mut |HeaderLine { revision, directive, .. }| {
                // We're trying to *find* `//@ revision: xxx` directives themselves, not revisioned
                // directives.
                if revision.is_some() {
                    return;
                }

                let directive = directive.trim();

                if directive.starts_with("revisions") {
                    let Some((name, value)) = directive.split_once([':', ' ']) else {
                        return;
                    };

                    if name == "revisions" {
                        let revs = value.split(' ');
                        for rev in revs {
                            expected_revisions.insert(rev.to_owned());
                        }
                    }
                }
            });

            let Some(test_name) = test.file_stem().and_then(OsStr::to_str) else {
                continue;
            };

            assert!(
                !test_name.contains('.'),
                "test name cannot contain dots '.': `{}`",
                test.display()
            );

            test_info.insert(test_name.to_string(), (test, expected_revisions));
        }

        // Our test file `foo.rs` has specified no revisions. There should not be any
        // `foo.rev{.stderr,.stdout}` files. rustc-dev-guide says test output files can have names
        // of the form: `test-name.revision.compare_mode.extension`, but our only concern is
        // `test-name.revision` and `extension`.
        for sibling in files_under_inspection.iter().filter(|f| {
            f.extension().and_then(OsStr::to_str).is_some_and(|ext| EXTENSIONS.contains(&ext))
        }) {
            let Some(filename) = sibling.file_name().and_then(OsStr::to_str) else {
                continue;
            };

            let filename_components = filename.split('.').collect::<Vec<_>>();
            let [file_prefix, ..] = &filename_components[..] else {
                continue;
            };

            let Some((test_path, expected_revisions)) = test_info.get(*file_prefix) else {
                continue;
            };

            match &filename_components[..] {
                // Cannot have a revision component, skip.
                [] | [_] => return,
                [_, _] if !expected_revisions.is_empty() => {
                    // Found unrevisioned output files for a revisioned test.
                    tidy_error!(
                        bad,
                        "found unrevisioned output file `{}` for a revisioned test `{}`",
                        sibling.display(),
                        test_path.display(),
                    );
                }
                [_, _] => return,
                [_, found_revision, .., extension] => {
                    if !IGNORES.contains(found_revision)
                        && !expected_revisions.contains(*found_revision)
                        // This is from `//@ stderr-per-bitwidth`
                        && !(*extension == "stderr" && ["32bit", "64bit"].contains(found_revision))
                    {
                        // Found some unexpected revision-esque component that is not a known
                        // compare-mode or expected revision.
                        tidy_error!(
                            bad,
                            "found output file `{}` for unexpected revision `{}` of test `{}`",
                            sibling.display(),
                            found_revision,
                            test_path.display()
                        );
                    }
                }
            }
        }
    });
}

fn filter(path: &Path) -> bool {
    filter_dirs(path) // ignore certain dirs
        || (path.file_name().is_some_and(|name| name == "auxiliary")) // ignore auxiliary folder
}
