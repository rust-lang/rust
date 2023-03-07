//! Tidy check to prevent creation of unnecessary debug artifacts while running tests.

use crate::walk::{filter_dirs, walk};
use std::path::Path;

const GRAPHVIZ_POSTFLOW_MSG: &str = "`borrowck_graphviz_postflow` attribute in test";

pub fn check(test_dir: &Path, bad: &mut bool) {
    walk(test_dir, filter_dirs, &mut |entry, contents| {
        let filename = entry.path();
        let is_rust = filename.extension().map_or(false, |ext| ext == "rs");
        if !is_rust {
            return;
        }

        for (i, line) in contents.lines().enumerate() {
            if line.contains("borrowck_graphviz_postflow") {
                tidy_error!(bad, "{}:{}: {}", filename.display(), i + 1, GRAPHVIZ_POSTFLOW_MSG);
            }
        }
    });
}
