//! Tidy check to ensure that crate `edition` is '2018' or '2021'.

use std::path::Path;

use crate::walk::{filter_dirs, walk};

pub fn check(path: &Path, bad: &mut bool) {
    walk(path, |path, _is_dir| filter_dirs(path), &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap();
        if filename != "Cargo.toml" {
            return;
        }

        let is_2021 = contents.lines().any(|line| line.trim() == "edition = \"2021\"");

        let is_workspace = contents.lines().any(|line| line.trim() == "[workspace]");
        let is_package = contents.lines().any(|line| line.trim() == "[package]");
        assert!(is_workspace || is_package);

        // Check that all packages use the 2021 edition. Virtual workspaces don't allow setting an
        // edition, so these shouldn't be checked.
        if is_package && !is_2021 {
            tidy_error!(
                bad,
                "{} doesn't have `edition = \"2021\"` on a separate line",
                file.display()
            );
        }
    });
}
