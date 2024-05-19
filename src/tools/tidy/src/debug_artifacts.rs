//! Tidy check to prevent creation of unnecessary debug artifacts while running tests.

use crate::walk::{filter_dirs, filter_not_rust, walk};
use std::path::Path;

const GRAPHVIZ_POSTFLOW_MSG: &str = "`borrowck_graphviz_postflow` attribute in test";

pub fn check(test_dir: &Path, bad: &mut bool) {
    walk(
        test_dir,
        |path, _is_dir| filter_dirs(path) || filter_not_rust(path),
        &mut |entry, contents| {
            for (i, line) in contents.lines().enumerate() {
                if line.contains("borrowck_graphviz_postflow") {
                    tidy_error!(
                        bad,
                        "{}:{}: {}",
                        entry.path().display(),
                        i + 1,
                        GRAPHVIZ_POSTFLOW_MSG
                    );
                }
            }
        },
    );
}
