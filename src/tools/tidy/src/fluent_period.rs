//! Checks that no Fluent messages or attributes end in periods (except ellipses)

use std::path::Path;

use fluent_syntax::ast::{Entry, PatternElement};

use crate::diagnostics::{CheckId, RunningCheck, TidyCtx};
use crate::walk::{filter_dirs, walk};

fn filter_fluent(path: &Path) -> bool {
    if let Some(ext) = path.extension() { ext.to_str() != Some("ftl") } else { true }
}

/// Messages allowed to have `.` at their end.
///
/// These should probably be reworked eventually.
const ALLOWLIST: &[&str] = &[
    "const_eval_long_running",
    "const_eval_validation_failure_note",
    "driver_impl_ice",
    "incremental_corrupt_file",
];

fn check_period(filename: &str, contents: &str, check: &mut RunningCheck) {
    if filename.contains("codegen") {
        // FIXME: Too many codegen messages have periods right now...
        return;
    }

    let (Ok(parse) | Err((parse, _))) = fluent_syntax::parser::parse(contents);
    for entry in &parse.body {
        if let Entry::Message(m) = entry {
            if ALLOWLIST.contains(&m.id.name) {
                continue;
            }

            if let Some(pat) = &m.value
                && let Some(PatternElement::TextElement { value }) = pat.elements.last()
            {
                // We don't care about ellipses.
                if value.ends_with(".") && !value.ends_with("...") {
                    let ll = find_line(contents, value);
                    let name = m.id.name;
                    check.error(format!("{filename}:{ll}: message `{name}` ends in a period"));
                }
            }

            for attr in &m.attributes {
                // Teach notes usually have long messages.
                if attr.id.name == "teach_note" {
                    continue;
                }

                if let Some(PatternElement::TextElement { value }) = attr.value.elements.last()
                    && value.ends_with(".")
                    && !value.ends_with("...")
                {
                    let ll = find_line(contents, value);
                    let name = attr.id.name;
                    check.error(format!("{filename}:{ll}: attr `{name}` ends in a period"));
                }
            }
        }
    }
}

/// Evil cursed bad hack. Requires that `value` be a substr (in memory) of `contents`.
fn find_line(haystack: &str, needle: &str) -> usize {
    for (ll, line) in haystack.lines().enumerate() {
        if line.as_ptr() > needle.as_ptr() {
            return ll;
        }
    }

    1
}

pub fn check(path: &Path, tidy_ctx: TidyCtx) {
    let mut check = tidy_ctx.start_check(CheckId::new("fluent_period").path(path));

    walk(
        path,
        |path, is_dir| filter_dirs(path) || (!is_dir && filter_fluent(path)),
        &mut |ent, contents| {
            check_period(ent.path().to_str().unwrap(), contents, &mut check);
        },
    );
}
