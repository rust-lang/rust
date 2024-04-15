//! Tidy check to ensure that tests inside 'tests/crashes' have a '@known-bug' directive.

use crate::walk::*;
use std::path::Path;

pub fn check(filepath: &Path, bad: &mut bool) {
    walk(filepath, |path, _is_dir| filter_not_rust(path), &mut |entry, contents| {
        let file = entry.path();
        if !contents.lines().any(|line| line.starts_with("//@ known-bug: ")) {
            tidy_error!(
                bad,
                "{} crash/ice test does not have a \"//@ known-bug: \" directive",
                file.display()
            );
        }
    });
}
