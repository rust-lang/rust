//! These structs are a subset of the ones found in `rustc_errors::json`.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use serde::Deserialize;

use crate::errors::{Error, ErrorKind};

#[derive(Deserialize)]
struct Diagnostic {
    message: String,
    code: Option<DiagnosticCode>,
    level: String,
    spans: Vec<DiagnosticSpan>,
    children: Vec<Diagnostic>,
    rendered: Option<String>,
}

#[derive(Deserialize)]
struct ArtifactNotification {
    #[allow(dead_code)]
    artifact: PathBuf,
}

#[derive(Deserialize)]
struct UnusedExternNotification {
    #[allow(dead_code)]
    lint_level: String,
    #[allow(dead_code)]
    unused_extern_names: Vec<String>,
}

#[derive(Deserialize, Clone)]
struct DiagnosticSpan {
    file_name: String,
    line_start: usize,
    column_start: usize,
    is_primary: bool,
    label: Option<String>,
    suggested_replacement: Option<String>,
    expansion: Option<Box<DiagnosticSpanMacroExpansion>>,
}

#[derive(Deserialize)]
struct FutureIncompatReport {
    future_incompat_report: Vec<FutureBreakageItem>,
}

#[derive(Deserialize)]
struct FutureBreakageItem {
    diagnostic: Diagnostic,
}

impl DiagnosticSpan {
    /// Returns the deepest source span in the macro call stack with a given file name.
    /// This is either the supplied span, or the span for some macro callsite that expanded to it.
    fn first_callsite_in_file(&self, file_name: &str) -> &DiagnosticSpan {
        if self.file_name == file_name {
            self
        } else {
            self.expansion
                .as_ref()
                .map(|origin| origin.span.first_callsite_in_file(file_name))
                .unwrap_or(self)
        }
    }
}

#[derive(Deserialize, Clone)]
struct DiagnosticSpanMacroExpansion {
    /// span where macro was applied to generate this code
    span: DiagnosticSpan,

    /// name of macro that was applied (e.g., "foo!" or "#[derive(Eq)]")
    macro_decl_name: String,
}

#[derive(Deserialize, Clone)]
struct DiagnosticCode {
    /// The code itself.
    code: String,
}

pub(crate) fn rustfix_diagnostics_only(output: &str) -> String {
    output
        .lines()
        .filter(|line| line.starts_with('{') && serde_json::from_str::<Diagnostic>(line).is_ok())
        .collect()
}

/// Extracts the rendered diagnostics from compiler json output.
///
/// When `sort` is set the diagnostics are reordered by their primary source
/// location. This is used under the parallel front-end, where diagnostics are
/// emitted in a nondeterministic order: the alloc-id normalization in `runtest`
/// assigns `ALLOC<n>` indices by first appearance in this text, so an unstable
/// order would otherwise produce unstable numbering. Sorting by source location
/// makes the numbering deterministic without changing single-threaded output.
pub(crate) fn extract_rendered(output: &str, sort: bool) -> String {
    let items = output.lines().filter_map(|line| {
        if line.starts_with('{') {
            if let Ok(diagnostic) = serde_json::from_str::<Diagnostic>(line) {
                let primary = diagnostic
                    .spans
                    .iter()
                    .find(|span| span.is_primary)
                    .map(|span| (span.file_name.clone(), span.line_start, span.column_start));
                diagnostic.rendered.map(|text| (primary, text))
            } else if let Ok(report) = serde_json::from_str::<FutureIncompatReport>(line) {
                if report.future_incompat_report.is_empty() {
                    None
                } else {
                    Some((
                        None,
                        format!(
                            "Future incompatibility report: {}",
                            report
                                .future_incompat_report
                                .into_iter()
                                .map(|item| {
                                    format!(
                                        "Future breakage diagnostic:\n{}",
                                        item.diagnostic
                                            .rendered
                                            .unwrap_or_else(|| "Not rendered".to_string())
                                    )
                                })
                                .collect::<String>()
                        ),
                    ))
                }
            } else if serde_json::from_str::<ArtifactNotification>(line).is_ok() {
                // Ignore the notification.
                None
            } else if serde_json::from_str::<UnusedExternNotification>(line).is_ok() {
                // Ignore the notification.
                None
            } else {
                // This function is called for both compiler and non-compiler output,
                // so if the line isn't recognized as JSON from the compiler then
                // just print it as-is.
                Some((None, format!("{line}\n")))
            }
        } else {
            // preserve non-JSON lines, such as ICEs
            Some((None, format!("{}\n", line)))
        }
    });

    if !sort {
        return items.map(|(_, text)| text).collect();
    }

    // Order located diagnostics by source position; ties broken by their shape
    // (with concrete alloc ids blinded, since those are racy under `-Zthreads`).
    // Items without a primary span keep their original relative order and follow
    // the located diagnostics, so the trailing "aborting due to N errors" summary
    // stays last.
    let mut items: Vec<_> = items.collect();
    items.sort_by(|(a_key, a_text), (b_key, b_text)| match (a_key, b_key) {
        (Some(a_key), Some(b_key)) => {
            a_key.cmp(b_key).then_with(|| alloc_blind(a_text).cmp(&alloc_blind(b_text)))
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });
    items.into_iter().map(|(_, text)| text).collect()
}

/// Replaces concrete allocation ids with a placeholder so that two diagnostics
/// at the same source location sort deterministically by their shape rather than
/// by the raw allocation number, which is racy under the parallel front-end.
fn alloc_blind(rendered: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\ba(?:lloc)?\d+\b").unwrap());
    re.replace_all(rendered, "alloc").into_owned()
}

pub(crate) fn parse_output(file_name: &str, output: &str) -> Vec<Error> {
    let mut errors = Vec::new();
    for line in output.lines() {
        // Compiler can emit non-json lines in non-`--error-format=json` modes,
        // and in some situations even in json mode.
        match serde_json::from_str::<Diagnostic>(line) {
            Ok(diagnostic) => push_actual_errors(&mut errors, &diagnostic, &[], file_name),
            Err(_) => errors.push(Error {
                line_num: None,
                column_num: None,
                kind: ErrorKind::Raw,
                msg: line.to_string(),
                require_annotation: false,
            }),
        }
    }
    errors
}

fn push_actual_errors(
    errors: &mut Vec<Error>,
    diagnostic: &Diagnostic,
    default_spans: &[&DiagnosticSpan],
    file_name: &str,
) {
    // In case of macro expansions, we need to get the span of the callsite
    let spans_info_in_this_file: Vec<_> = diagnostic
        .spans
        .iter()
        .map(|span| (span.is_primary, span.first_callsite_in_file(file_name)))
        .filter(|(_, span)| Path::new(&span.file_name) == Path::new(&file_name))
        .collect();

    let primary_spans: Vec<_> = spans_info_in_this_file
        .iter()
        .filter(|(is_primary, _)| *is_primary)
        .map(|(_, span)| span)
        .take(1) // sometimes we have more than one showing up in the json; pick first
        .cloned()
        .collect();
    let primary_spans = if primary_spans.is_empty() {
        // subdiagnostics often don't have a span of their own;
        // inherit the span from the parent in that case
        default_spans
    } else {
        &primary_spans
    };

    // We break the output into multiple lines, and then append the
    // [E123] to every line in the output. This may be overkill.  The
    // intention was to match existing tests that do things like "//|
    // found `i32` [E123]" and expect to match that somewhere, and yet
    // also ensure that `//~ ERROR E123` *always* works. The
    // assumption is that these multi-line error messages are on their
    // way out anyhow.
    let with_code = |text| match &diagnostic.code {
        Some(code) => format!("{text} [{}]", code.code),
        None => format!("{text}"),
    };

    // Convert multi-line messages into multiple errors.
    // We expect to replace these with something more structured anyhow.
    let mut message_lines = diagnostic.message.lines();
    let kind = ErrorKind::from_compiler_str(&diagnostic.level);
    let first_line = message_lines.next().unwrap_or(&diagnostic.message);
    if primary_spans.is_empty() {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re_init =
            || Regex::new(r"aborting due to \d+ previous errors?|\d+ warnings? emitted").unwrap();
        errors.push(Error {
            line_num: None,
            column_num: None,
            kind,
            msg: with_code(first_line),
            require_annotation: diagnostic.level != "failure-note"
                && !RE.get_or_init(re_init).is_match(first_line),
        });
    } else {
        for span in primary_spans {
            errors.push(Error {
                line_num: Some(span.line_start),
                column_num: Some(span.column_start),
                kind,
                msg: with_code(first_line),
                require_annotation: true,
            });
        }
    }
    for next_line in message_lines {
        if primary_spans.is_empty() {
            errors.push(Error {
                line_num: None,
                column_num: None,
                kind,
                msg: with_code(next_line),
                require_annotation: false,
            });
        } else {
            for span in primary_spans {
                errors.push(Error {
                    line_num: Some(span.line_start),
                    column_num: Some(span.column_start),
                    kind,
                    msg: with_code(next_line),
                    require_annotation: false,
                });
            }
        }
    }

    // If the message has a suggestion, register that.
    for span in primary_spans {
        if let Some(ref suggested_replacement) = span.suggested_replacement {
            for (index, line) in suggested_replacement.lines().enumerate() {
                errors.push(Error {
                    line_num: Some(span.line_start + index),
                    column_num: Some(span.column_start),
                    kind: ErrorKind::Suggestion,
                    msg: line.to_string(),
                    // Empty suggestions (suggestions to remove something) are common
                    // and annotating them in source is not useful.
                    require_annotation: !line.is_empty(),
                });
            }
        }
    }

    // Add notes for the backtrace
    for span in primary_spans {
        if let Some(frame) = &span.expansion {
            push_backtrace(errors, frame, file_name);
        }
    }

    // Add notes for any labels that appear in the message.
    for (_, span) in spans_info_in_this_file {
        if let Some(label) = &span.label {
            errors.push(Error {
                line_num: Some(span.line_start),
                column_num: Some(span.column_start),
                kind: ErrorKind::Note,
                msg: label.clone(),
                // Empty labels (only underlining spans) are common and do not need annotations.
                require_annotation: !label.is_empty(),
            });
        }
    }

    // Flatten out the children.
    for child in &diagnostic.children {
        push_actual_errors(errors, child, primary_spans, file_name);
    }
}

fn push_backtrace(
    errors: &mut Vec<Error>,
    expansion: &DiagnosticSpanMacroExpansion,
    file_name: &str,
) {
    if Path::new(&expansion.span.file_name) == Path::new(&file_name) {
        errors.push(Error {
            line_num: Some(expansion.span.line_start),
            column_num: Some(expansion.span.column_start),
            kind: ErrorKind::Note,
            msg: format!("in this expansion of {}", expansion.macro_decl_name),
            require_annotation: true,
        });
    }

    if let Some(previous_expansion) = &expansion.span.expansion {
        push_backtrace(errors, previous_expansion, file_name);
    }
}
