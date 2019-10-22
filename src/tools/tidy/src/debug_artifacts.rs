//! Tidy check to prevent creation of unnecessary debug artifacts.

use std::path::{Path, PathBuf};

const GRAPHVIZ_POSTFLOW_MSG: &'static str =
    "`borrowck_graphviz_postflow` attribute in test";

pub fn check(path: &Path, bad: &mut bool) {
    let test_dir: PathBuf = path.join("test");

    super::walk(&test_dir, &mut super::filter_dirs, &mut |entry, contents| {
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
