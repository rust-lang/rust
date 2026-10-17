//! Tidy check to ensure that rustdoc GUI tests start with a small description.

use std::path::Path;

use crate::diagnostics::{CheckId, TidyCtx};

pub fn check(path: &Path, tidy_ctx: TidyCtx) {
    let mut check = tidy_ctx.start_check(CheckId::new("rustdoc_gui_tests").path(path));

    crate::walk::walk(
        &path.join("rustdoc-gui"),
        |p, is_dir| !is_dir && p.extension().is_none_or(|e| e != "goml"),
        &mut |entry, content| {
            for line in content.lines() {
                if !line.starts_with("// ") {
                    check.error(format!(
                        "{}: rustdoc-gui tests must start with a small description",
                        entry.path().display(),
                    ));
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
