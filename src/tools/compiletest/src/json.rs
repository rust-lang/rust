//! These structs are a subset of the ones found in `rustc_errors::json`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use serde::Deserialize;

use crate::errors::{Error, ErrorKind};
use crate::runtest::ProcRes;

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
    line_end: usize,
    column_start: usize,
    column_end: usize,
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

pub fn rustfix_diagnostics_only(output: &str) -> String {
    output
        .lines()
        .filter(|line| line.starts_with('{') && serde_json::from_str::<Diagnostic>(line).is_ok())
        .collect()
}

pub fn extract_rendered(output: &str) -> String {
    output
        .lines()
        .filter_map(|line| {
            if line.starts_with('{') {
                if let Ok(diagnostic) = serde_json::from_str::<Diagnostic>(line) {
                    diagnostic.rendered
                } else if let Ok(report) = serde_json::from_str::<FutureIncompatReport>(line) {
                    if report.future_incompat_report.is_empty() {
                        None
                    } else {
                        Some(format!(
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
                    Some(format!("{line}\n"))
                }
            } else {
                // preserve non-JSON lines, such as ICEs
                Some(format!("{}\n", line))
            }
        })
        .collect()
}

pub fn parse_output(file_name: &str, output: &str, proc_res: &ProcRes) -> Vec<Error> {
    let mut errors = Vec::new();
    for line in output.lines() {
        // The compiler sometimes intermingles non-JSON stuff into the
        // output.  This hack just skips over such lines. Yuck.
        if line.starts_with('{') {
            match serde_json::from_str::<Diagnostic>(line) {
                Ok(diagnostic) => push_actual_errors(&mut errors, &diagnostic, &[], file_name),
                Err(error) => {
                    // Ignore the future compat report message - this is handled
                    // by `extract_rendered`
                    if serde_json::from_str::<FutureIncompatReport>(line).is_err() {
                        proc_res.fatal(
                        Some(&format!(
                            "failed to decode compiler output as json: `{}`\nline: {}\noutput: {}",
                            error, line, output
                        )),
                        || (),
                    );
                    }
                }
            }
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

    let spans_in_this_file: Vec<_> = spans_info_in_this_file.iter().map(|(_, span)| span).collect();

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
    let with_code = |span: Option<&DiagnosticSpan>, text: &str| {
        // FIXME(#33000) -- it'd be better to use a dedicated
        // UI harness than to include the line/col number like
        // this, but some current tests rely on it.
        //
        // Note: Do NOT include the filename. These can easily
        // cause false matches where the expected message
        // appears in the filename, and hence the message
        // changes but the test still passes.
        let span_str = match span {
            Some(DiagnosticSpan { line_start, column_start, line_end, column_end, .. }) => {
                format!("{line_start}:{column_start}: {line_end}:{column_end}")
            }
            None => format!("?:?: ?:?"),
        };
        match &diagnostic.code {
            Some(code) => format!("{span_str}: {text} [{}]", code.code),
            None => format!("{span_str}: {text}"),
        }
    };

    // Convert multi-line messages into multiple errors.
    // We expect to replace these with something more structured anyhow.
    let mut message_lines = diagnostic.message.lines();
    let kind = Some(ErrorKind::from_compiler_str(&diagnostic.level));
    let first_line = message_lines.next().unwrap_or(&diagnostic.message);
    if primary_spans.is_empty() {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re_init =
            || Regex::new(r"aborting due to \d+ previous errors?|\d+ warnings? emitted").unwrap();
        errors.push(Error {
            line_num: None,
            kind,
            msg: with_code(None, first_line),
            require_annotation: diagnostic.level != "failure-note"
                && !RE.get_or_init(re_init).is_match(first_line),
        });
    } else {
        for span in primary_spans {
            errors.push(Error {
                line_num: Some(span.line_start),
                kind,
                msg: with_code(Some(span), first_line),
                require_annotation: true,
            });
        }
    }
    for next_line in message_lines {
        if primary_spans.is_empty() {
            errors.push(Error {
                line_num: None,
                kind,
                msg: with_code(None, next_line),
                require_annotation: false,
            });
        } else {
            for span in primary_spans {
                errors.push(Error {
                    line_num: Some(span.line_start),
                    kind,
                    msg: with_code(Some(span), next_line),
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
                    kind: Some(ErrorKind::Suggestion),
                    msg: line.to_string(),
                    require_annotation: true,
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
    for span in spans_in_this_file.iter().filter(|span| span.label.is_some()) {
        errors.push(Error {
            line_num: Some(span.line_start),
            kind: Some(ErrorKind::Note),
            msg: span.label.clone().unwrap(),
            require_annotation: true,
        });
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
            kind: Some(ErrorKind::Note),
            msg: format!("in this expansion of {}", expansion.macro_decl_name),
            require_annotation: true,
        });
    }

    if let Some(previous_expansion) = &expansion.span.expansion {
        push_backtrace(errors, previous_expansion, file_name);
    }
}
