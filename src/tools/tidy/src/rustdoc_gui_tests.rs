//! Tidy check to ensure that rustdoc GUI tests start with a small description.

use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    crate::walk::walk(
        &path.join("rustdoc-gui"),
        |p| {
            // If there is no extension, it's very likely a folder and we want to go into it.
            p.extension().map(|e| e != "goml").unwrap_or(false)
        },
        &mut |entry, content| {
            for line in content.lines() {
                if !line.starts_with("// ") {
                    tidy_error!(
                        bad,
                        "{}: rustdoc-gui tests must start with a small description",
                        entry.path().display(),
                    );
                    return;
                } else if line.starts_with("// ") {
                    let parts = line[2..].trim();
                    // We ignore tidy comments.
                    if parts.starts_with("// tidy-") {
                        continue;
                    }
                    // All good!
                    return;
                }
            }
        },
    );
}
