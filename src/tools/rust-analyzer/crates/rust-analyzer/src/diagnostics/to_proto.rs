//! This module provides the functionality needed to convert diagnostics from
//! `cargo check` json format to the LSP diagnostic format.

use crate::flycheck::{Applicability, DiagnosticLevel, DiagnosticSpan};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use stdx::format_to;
use vfs::{AbsPath, AbsPathBuf};

use crate::{
    global_state::GlobalStateSnapshot, line_index::PositionEncoding,
    lsp::to_proto::url_from_abs_path, lsp_ext,
};

use super::{DiagnosticsMapConfig, Fix};

/// Determines the LSP severity from a diagnostic
fn diagnostic_severity(
    config: &DiagnosticsMapConfig,
    level: crate::flycheck::DiagnosticLevel,
    code: Option<crate::flycheck::DiagnosticCode>,
) -> Option<lsp_types::DiagnosticSeverity> {
    let res = match level {
        DiagnosticLevel::Ice => lsp_types::DiagnosticSeverity::ERROR,
        DiagnosticLevel::Error => lsp_types::DiagnosticSeverity::ERROR,
        DiagnosticLevel::Warning => match &code {
            // HACK: special case for `warnings` rustc lint.
            Some(code)
                if config.warnings_as_hint.iter().any(|lint| {
                    lint == "warnings" || ide_db::helpers::lint_eq_or_in_group(&code.code, lint)
                }) =>
            {
                lsp_types::DiagnosticSeverity::HINT
            }
            // HACK: special case for `warnings` rustc lint.
            Some(code)
                if config.warnings_as_info.iter().any(|lint| {
                    lint == "warnings" || ide_db::helpers::lint_eq_or_in_group(&code.code, lint)
                }) =>
            {
                lsp_types::DiagnosticSeverity::INFORMATION
            }
            _ => lsp_types::DiagnosticSeverity::WARNING,
        },
        DiagnosticLevel::Note => lsp_types::DiagnosticSeverity::INFORMATION,
        DiagnosticLevel::Help => lsp_types::DiagnosticSeverity::HINT,
        _ => return None,
    };
    Some(res)
}

/// Checks whether a file name is from macro invocation and does not refer to an actual file.
fn is_dummy_macro_file(file_name: &str) -> bool {
    // FIXME: current rustc does not seem to emit `<macro file>` files anymore?
    file_name.starts_with('<') && file_name.ends_with('>')
}

/// Converts a Rust span to a LSP location
fn location(
    config: &DiagnosticsMapConfig,
    workspace_root: &AbsPath,
    span: &DiagnosticSpan,
    snap: &GlobalStateSnapshot,
) -> lsp_types::Location {
    let file_name = resolve_path(config, workspace_root, &span.file_name);
    let uri = url_from_abs_path(&file_name);

    let range = {
        let position_encoding = snap.config.negotiated_encoding();
        lsp_types::Range::new(
            position(
                &position_encoding,
                span,
                span.line_start,
                span.column_start.saturating_sub(1),
            ),
            position(&position_encoding, span, span.line_end, span.column_end.saturating_sub(1)),
        )
    };
    lsp_types::Location::new(uri, range)
}

fn position(
    position_encoding: &PositionEncoding,
    span: &DiagnosticSpan,
    line_number: usize,
    column_offset_utf32: usize,
) -> lsp_types::Position {
    let line_index = line_number - span.line_start;

    let column_offset_encoded = match span.text.get(line_index) {
        // Fast path.
        Some(line) if line.text.is_ascii() => column_offset_utf32,
        Some(line) => {
            let line_prefix_len = line
                .text
                .char_indices()
                .take(column_offset_utf32)
                .last()
                .map(|(pos, c)| pos + c.len_utf8())
                .unwrap_or(0);
            let line_prefix = &line.text[..line_prefix_len];
            match position_encoding {
                PositionEncoding::Utf8 => line_prefix.len(),
                PositionEncoding::Wide(enc) => enc.measure(line_prefix),
            }
        }
        None => column_offset_utf32,
    };

    lsp_types::Position {
        line: (line_number as u32).saturating_sub(1),
        character: column_offset_encoded as u32,
    }
}

/// Extracts a suitable "primary" location from a rustc diagnostic.
///
/// This takes locations pointing into the standard library, or generally outside the current
/// workspace into account and tries to avoid those, in case macros are involved.
fn primary_location(
    config: &DiagnosticsMapConfig,
    workspace_root: &AbsPath,
    span: &DiagnosticSpan,
    snap: &GlobalStateSnapshot,
) -> lsp_types::Location {
    let span_stack = std::iter::successors(Some(span), |span| Some(&span.expansion.as_ref()?.span));
    for span in span_stack.clone() {
        let abs_path = resolve_path(config, workspace_root, &span.file_name);
        if !is_dummy_macro_file(&span.file_name) && abs_path.starts_with(workspace_root) {
            return location(config, workspace_root, span, snap);
        }
    }

    // Fall back to the outermost macro invocation if no suitable span comes up.
    let last_span = span_stack.last().unwrap();
    location(config, workspace_root, last_span, snap)
}

/// Converts a secondary Rust span to a LSP related information
///
/// If the span is unlabelled this will return `None`.
fn diagnostic_related_information(
    config: &DiagnosticsMapConfig,
    workspace_root: &AbsPath,
    span: &DiagnosticSpan,
    snap: &GlobalStateSnapshot,
) -> Option<lsp_types::DiagnosticRelatedInformation> {
    let message = span.label.clone()?;
    let location = location(config, workspace_root, span, snap);
    Some(lsp_types::DiagnosticRelatedInformation { location, message })
}

/// Resolves paths applying any matching path prefix remappings, and then
/// joining the path to the workspace root.
fn resolve_path(
    config: &DiagnosticsMapConfig,
    workspace_root: &AbsPath,
    file_name: &str,
) -> AbsPathBuf {
    match config
        .remap_prefix
        .iter()
        .find_map(|(from, to)| file_name.strip_prefix(from).map(|file_name| (to, file_name)))
    {
        Some((to, file_name)) => workspace_root.join(format!("{to}{file_name}")),
        None => workspace_root.join(file_name),
    }
}

struct SubDiagnostic {
    related: lsp_types::DiagnosticRelatedInformation,
    suggested_fix: Option<Box<Fix>>,
}

enum MappedRustChildDiagnostic {
    SubDiagnostic(SubDiagnostic),
    MessageLine(String),
}

fn map_rust_child_diagnostic(
    config: &DiagnosticsMapConfig,
    workspace_root: &AbsPath,
    rd: &crate::flycheck::Diagnostic,
    snap: &GlobalStateSnapshot,
) -> MappedRustChildDiagnostic {
    let spans: Vec<&DiagnosticSpan> = rd.spans.iter().filter(|s| s.is_primary).collect();
    if spans.is_empty() {
        // `rustc` uses these spanless children as a way to print multi-line
        // messages
        return MappedRustChildDiagnostic::MessageLine(rd.message.clone());
    }

    let mut edit_map: FxHashMap<lsp_types::Url, Vec<lsp_types::TextEdit>> = FxHashMap::default();
    let mut suggested_replacements = Vec::new();
    let mut is_preferred = true;
    for &span in &spans {
        if let Some(suggested_replacement) = &span.suggested_replacement {
            if !suggested_replacement.is_empty() {
                suggested_replacements.push(suggested_replacement);
            }
            let location = location(config, workspace_root, span, snap);
            let edit = lsp_types::TextEdit::new(location.range, suggested_replacement.clone());

            // Only actually emit a quickfix if the suggestion is "valid enough".
            // We accept both "MaybeIncorrect" and "MachineApplicable". "MaybeIncorrect" means that
            // the suggestion is *complete* (contains no placeholders where code needs to be
            // inserted), but might not be what the user wants, or might need minor adjustments.
            if matches!(
                span.suggestion_applicability,
                None | Some(Applicability::MaybeIncorrect | Applicability::MachineApplicable)
            ) {
                edit_map.entry(location.uri).or_default().push(edit);
            }
            is_preferred &=
                matches!(span.suggestion_applicability, Some(Applicability::MachineApplicable));
        }
    }

    // rustc renders suggestion diagnostics by appending the suggested replacement, so do the same
    // here, otherwise the diagnostic text is missing useful information.
    let mut message = rd.message.clone();
    if !suggested_replacements.is_empty() {
        message.push_str(": ");
        let suggestions =
            suggested_replacements.iter().map(|suggestion| format!("`{suggestion}`")).join(", ");
        message.push_str(&suggestions);
    }

    if edit_map.is_empty() {
        MappedRustChildDiagnostic::SubDiagnostic(SubDiagnostic {
            related: lsp_types::DiagnosticRelatedInformation {
                location: location(config, workspace_root, spans[0], snap),
                message,
            },
            suggested_fix: None,
        })
    } else {
        MappedRustChildDiagnostic::SubDiagnostic(SubDiagnostic {
            related: lsp_types::DiagnosticRelatedInformation {
                location: location(config, workspace_root, spans[0], snap),
                message: message.clone(),
            },
            suggested_fix: Some(Box::new(Fix {
                ranges: spans
                    .iter()
                    .map(|&span| location(config, workspace_root, span, snap).range)
                    .collect(),
                action: lsp_ext::CodeAction {
                    title: message,
                    group: None,
                    kind: Some(lsp_types::CodeActionKind::QUICKFIX),
                    edit: Some(lsp_ext::SnippetWorkspaceEdit {
                        // FIXME: there's no good reason to use edit_map here....
                        changes: Some(edit_map),
                        document_changes: None,
                        change_annotations: None,
                    }),
                    is_preferred: Some(is_preferred),
                    data: None,
                    command: None,
                },
            })),
        })
    }
}

#[derive(Debug)]
pub(crate) struct MappedRustDiagnostic {
    pub(crate) url: lsp_types::Url,
    pub(crate) diagnostic: lsp_types::Diagnostic,
    pub(crate) fix: Option<Box<Fix>>,
}

/// Converts a Rust root diagnostic to LSP form
///
/// This flattens the Rust diagnostic by:
///
/// 1. Creating a LSP diagnostic with the root message and primary span.
/// 2. Adding any labelled secondary spans to `relatedInformation`
/// 3. Categorising child diagnostics as either `SuggestedFix`es,
///    `relatedInformation` or additional message lines.
///
/// If the diagnostic has no primary span this will return `None`
pub(crate) fn map_rust_diagnostic_to_lsp(
    config: &DiagnosticsMapConfig,
    rd: &crate::flycheck::Diagnostic,
    workspace_root: &AbsPath,
    snap: &GlobalStateSnapshot,
) -> Vec<MappedRustDiagnostic> {
    let primary_spans: Vec<&DiagnosticSpan> = rd.spans.iter().filter(|s| s.is_primary).collect();
    if primary_spans.is_empty() {
        return Vec::new();
    }

    let severity = diagnostic_severity(config, rd.level, rd.code.clone());

    let mut source = String::from("rustc");
    let mut code = rd.code.as_ref().map(|c| c.code.clone());

    if let Some(code_val) = &code
        && config.check_ignore.contains(code_val)
    {
        return Vec::new();
    }

    if let Some(code_val) = &code {
        // See if this is an RFC #2103 scoped lint (e.g. from Clippy)
        let scoped_code: Vec<&str> = code_val.split("::").collect();
        if scoped_code.len() == 2 {
            source = String::from(scoped_code[0]);
            code = Some(String::from(scoped_code[1]));
        }
    }

    let mut needs_primary_span_label = true;
    let mut subdiagnostics = Vec::new();
    let mut tags = Vec::new();

    for secondary_span in rd.spans.iter().filter(|s| !s.is_primary) {
        let related = diagnostic_related_information(config, workspace_root, secondary_span, snap);
        if let Some(related) = related {
            subdiagnostics.push(SubDiagnostic { related, suggested_fix: None });
        }
    }

    let mut message = rd.message.clone();
    for child in &rd.children {
        let child = map_rust_child_diagnostic(config, workspace_root, child, snap);
        match child {
            MappedRustChildDiagnostic::SubDiagnostic(sub) => {
                subdiagnostics.push(sub);
            }
            MappedRustChildDiagnostic::MessageLine(message_line) => {
                format_to!(message, "\n{}", message_line);

                // These secondary messages usually duplicate the content of the
                // primary span label.
                needs_primary_span_label = false;
            }
        }
    }

    if let Some(code) = &rd.code {
        let code = code.code.as_str();
        if matches!(
            code,
            "dead_code"
                | "unknown_lints"
                | "unreachable_code"
                | "unused_attributes"
                | "unused_imports"
                | "unused_macros"
                | "unused_variables"
        ) {
            tags.push(lsp_types::DiagnosticTag::UNNECESSARY);
        }

        if matches!(code, "deprecated") {
            tags.push(lsp_types::DiagnosticTag::DEPRECATED);
        }
    }

    let code_description = match source.as_str() {
        "rustc" => rustc_code_description(code.as_deref()),
        "clippy" => clippy_code_description(code.as_deref()),
        _ => None,
    };

    primary_spans
        .iter()
        .flat_map(|primary_span| {
            let primary_location = primary_location(config, workspace_root, primary_span, snap);
            let message = {
                let mut message = message.clone();
                if needs_primary_span_label && let Some(primary_span_label) = &primary_span.label {
                    format_to!(message, "\n{}", primary_span_label);
                }
                message
            };
            // Each primary diagnostic span may result in multiple LSP diagnostics.
            let mut diagnostics = Vec::new();

            let mut related_info_macro_calls = vec![];

            // If error occurs from macro expansion, add related info pointing to
            // where the error originated
            // Also, we would generate an additional diagnostic, so that exact place of macro
            // will be highlighted in the error origin place.
            let span_stack = std::iter::successors(Some(*primary_span), |span| {
                Some(&span.expansion.as_ref()?.span)
            });
            for (i, span) in span_stack.enumerate() {
                if is_dummy_macro_file(&span.file_name) {
                    continue;
                }

                // First span is the original diagnostic, others are macro call locations that
                // generated that code.
                let is_in_macro_call = i != 0;

                let secondary_location = location(config, workspace_root, span, snap);
                if secondary_location == primary_location {
                    continue;
                }
                related_info_macro_calls.push(lsp_types::DiagnosticRelatedInformation {
                    location: secondary_location.clone(),
                    message: if is_in_macro_call {
                        "Error originated from macro call here".to_owned()
                    } else {
                        "Actual error occurred here".to_owned()
                    },
                });
                // For the additional in-macro diagnostic we add the inverse message pointing to the error location in code.
                let information_for_additional_diagnostic =
                    vec![lsp_types::DiagnosticRelatedInformation {
                        location: primary_location.clone(),
                        message: "Exact error occurred here".to_owned(),
                    }];

                let diagnostic = lsp_types::Diagnostic {
                    range: secondary_location.range,
                    // downgrade to hint if we're pointing at the macro
                    severity: Some(lsp_types::DiagnosticSeverity::HINT),
                    code: code.clone().map(lsp_types::NumberOrString::String),
                    code_description: code_description.clone(),
                    source: Some(source.clone()),
                    message: message.clone(),
                    related_information: Some(information_for_additional_diagnostic),
                    tags: if tags.is_empty() { None } else { Some(tags.clone()) },
                    data: Some(serde_json::json!({ "rendered": rd.rendered })),
                };
                diagnostics.push(MappedRustDiagnostic {
                    url: secondary_location.uri,
                    diagnostic,
                    fix: None,
                });
            }

            // Emit the primary diagnostic.
            diagnostics.push(MappedRustDiagnostic {
                url: primary_location.uri.clone(),
                diagnostic: lsp_types::Diagnostic {
                    range: primary_location.range,
                    severity,
                    code: code.clone().map(lsp_types::NumberOrString::String),
                    code_description: code_description.clone(),
                    source: Some(source.clone()),
                    message,
                    related_information: {
                        let info = related_info_macro_calls
                            .iter()
                            .cloned()
                            .chain(subdiagnostics.iter().map(|sub| sub.related.clone()))
                            .collect::<Vec<_>>();
                        if info.is_empty() { None } else { Some(info) }
                    },
                    tags: if tags.is_empty() { None } else { Some(tags.clone()) },
                    data: Some(serde_json::json!({ "rendered": rd.rendered })),
                },
                fix: None,
            });

            // Emit hint-level diagnostics for all `related_information` entries such as "help"s.
            // This is useful because they will show up in the user's editor, unlike
            // `related_information`, which just produces hard-to-read links, at least in VS Code.
            let back_ref = lsp_types::DiagnosticRelatedInformation {
                location: primary_location,
                message: "original diagnostic".to_owned(),
            };
            for sub in &subdiagnostics {
                diagnostics.push(MappedRustDiagnostic {
                    url: sub.related.location.uri.clone(),
                    fix: sub.suggested_fix.clone(),
                    diagnostic: lsp_types::Diagnostic {
                        range: sub.related.location.range,
                        severity: Some(lsp_types::DiagnosticSeverity::HINT),
                        code: code.clone().map(lsp_types::NumberOrString::String),
                        code_description: code_description.clone(),
                        source: Some(source.clone()),
                        message: sub.related.message.clone(),
                        related_information: Some(vec![back_ref.clone()]),
                        tags: None, // don't apply modifiers again
                        data: None,
                    },
                });
            }

            diagnostics
        })
        .collect()
}

fn rustc_code_description(code: Option<&str>) -> Option<lsp_types::CodeDescription> {
    code.filter(|code| {
        let mut chars = code.chars();
        chars.next() == Some('E')
            && chars.by_ref().take(4).all(|c| c.is_ascii_digit())
            && chars.next().is_none()
    })
    .and_then(|code| {
        lsp_types::Url::parse(&format!("https://doc.rust-lang.org/error-index.html#{code}"))
            .ok()
            .map(|href| lsp_types::CodeDescription { href })
    })
}

fn clippy_code_description(code: Option<&str>) -> Option<lsp_types::CodeDescription> {
    code.and_then(|code| {
        lsp_types::Url::parse(&format!(
            "https://rust-lang.github.io/rust-clippy/master/index.html#{code}"
        ))
        .ok()
        .map(|href| lsp_types::CodeDescription { href })
    })
}

#[cfg(test)]
#[cfg(not(windows))]
mod tests {
    use crate::{config::Config, global_state::GlobalState};

    use super::*;

    use expect_test::{ExpectFile, expect_file};
    use lsp_types::ClientCapabilities;
    use paths::Utf8Path;

    fn check(diagnostics_json: &str, expect: ExpectFile) {
        check_with_config(DiagnosticsMapConfig::default(), diagnostics_json, expect)
    }

    fn check_with_config(config: DiagnosticsMapConfig, diagnostics_json: &str, expect: ExpectFile) {
        let diagnostic: crate::flycheck::Diagnostic =
            serde_json::from_str(diagnostics_json).unwrap();
        let workspace_root: &AbsPath = Utf8Path::new("/test/").try_into().unwrap();
        let (sender, _) = crossbeam_channel::unbounded();
        let state = GlobalState::new(
            sender,
            Config::new(
                workspace_root.to_path_buf(),
                ClientCapabilities::default(),
                Vec::new(),
                None,
            ),
        );
        let snap = state.snapshot();
        let mut actual = map_rust_diagnostic_to_lsp(&config, &diagnostic, workspace_root, &snap);
        actual.iter_mut().for_each(|diag| diag.diagnostic.data = None);
        expect.assert_debug_eq(&actual)
    }

    #[test]
    fn rustc_incompatible_type_for_trait() {
        check(
            r##"{
                "message": "method `next` has an incompatible type for trait",
                "code": {
                    "code": "E0053",
                    "explanation": "\nThe parameters of any trait method must match between a trait implementation\nand the trait definition.\n\nHere are a couple examples of this error:\n\n```compile_fail,E0053\ntrait Foo {\n    fn foo(x: u16);\n    fn bar(&self);\n}\n\nstruct Bar;\n\nimpl Foo for Bar {\n    // error, expected u16, found i16\n    fn foo(x: i16) { }\n\n    // error, types differ in mutability\n    fn bar(&mut self) { }\n}\n```\n"
                },
                "level": "error",
                "spans": [
                    {
                        "file_name": "compiler/ty/list_iter.rs",
                        "byte_start": 1307,
                        "byte_end": 1350,
                        "line_start": 52,
                        "line_end": 52,
                        "column_start": 5,
                        "column_end": 48,
                        "is_primary": true,
                        "text": [
                            {
                                "text": "    fn next(&self) -> Option<&'list ty::Ref<M>> {",
                                "highlight_start": 5,
                                "highlight_end": 48
                            }
                        ],
                        "label": "types differ in mutability",
                        "suggested_replacement": null,
                        "suggestion_applicability": null,
                        "expansion": null
                    }
                ],
                "children": [
                    {
                        "message": "expected type `fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>`\n   found type `fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>`",
                        "code": null,
                        "level": "note",
                        "spans": [],
                        "children": [],
                        "rendered": null
                    }
                ],
                "rendered": "error[E0053]: method `next` has an incompatible type for trait\n  --> compiler/ty/list_iter.rs:52:5\n   |\n52 |     fn next(&self) -> Option<&'list ty::Ref<M>> {\n   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ types differ in mutability\n   |\n   = note: expected type `fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>`\n              found type `fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>`\n\n"
            }
            "##,
            expect_file!["./test_data/rustc_incompatible_type_for_trait.txt"],
        );
    }

    #[test]
    fn rustc_unused_variable() {
        check(
            r##"{
    "message": "unused variable: `foo`",
    "code": {
        "code": "unused_variables",
        "explanation": null
    },
    "level": "warning",
    "spans": [
        {
            "file_name": "driver/subcommand/repl.rs",
            "byte_start": 9228,
            "byte_end": 9231,
            "line_start": 291,
            "line_end": 291,
            "column_start": 9,
            "column_end": 12,
            "is_primary": true,
            "text": [
                {
                    "text": "    let foo = 42;",
                    "highlight_start": 9,
                    "highlight_end": 12
                }
            ],
            "label": null,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [
        {
            "message": "#[warn(unused_variables)] on by default",
            "code": null,
            "level": "note",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "consider prefixing with an underscore",
            "code": null,
            "level": "help",
            "spans": [
                {
                    "file_name": "driver/subcommand/repl.rs",
                    "byte_start": 9228,
                    "byte_end": 9231,
                    "line_start": 291,
                    "line_end": 291,
                    "column_start": 9,
                    "column_end": 12,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    let foo = 42;",
                            "highlight_start": 9,
                            "highlight_end": 12
                        }
                    ],
                    "label": null,
                    "suggested_replacement": "_foo",
                    "suggestion_applicability": "MachineApplicable",
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        }
    ],
    "rendered": "warning: unused variable: `foo`\n   --> driver/subcommand/repl.rs:291:9\n    |\n291 |     let foo = 42;\n    |         ^^^ help: consider prefixing with an underscore: `_foo`\n    |\n    = note: #[warn(unused_variables)] on by default\n\n"
    }"##,
            expect_file!["./test_data/rustc_unused_variable.txt"],
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn rustc_unused_variable_as_info() {
        check_with_config(
            DiagnosticsMapConfig {
                warnings_as_info: vec!["unused_variables".to_owned()],
                ..DiagnosticsMapConfig::default()
            },
            r##"{
    "message": "unused variable: `foo`",
    "code": {
        "code": "unused_variables",
        "explanation": null
    },
    "level": "warning",
    "spans": [
        {
            "file_name": "driver/subcommand/repl.rs",
            "byte_start": 9228,
            "byte_end": 9231,
            "line_start": 291,
            "line_end": 291,
            "column_start": 9,
            "column_end": 12,
            "is_primary": true,
            "text": [
                {
                    "text": "    let foo = 42;",
                    "highlight_start": 9,
                    "highlight_end": 12
                }
            ],
            "label": null,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [
        {
            "message": "#[warn(unused_variables)] on by default",
            "code": null,
            "level": "note",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "consider prefixing with an underscore",
            "code": null,
            "level": "help",
            "spans": [
                {
                    "file_name": "driver/subcommand/repl.rs",
                    "byte_start": 9228,
                    "byte_end": 9231,
                    "line_start": 291,
                    "line_end": 291,
                    "column_start": 9,
                    "column_end": 12,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    let foo = 42;",
                            "highlight_start": 9,
                            "highlight_end": 12
                        }
                    ],
                    "label": null,
                    "suggested_replacement": "_foo",
                    "suggestion_applicability": "MachineApplicable",
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        }
    ],
    "rendered": "warning: unused variable: `foo`\n   --> driver/subcommand/repl.rs:291:9\n    |\n291 |     let foo = 42;\n    |         ^^^ help: consider prefixing with an underscore: `_foo`\n    |\n    = note: #[warn(unused_variables)] on by default\n\n"
    }"##,
            expect_file!["./test_data/rustc_unused_variable_as_info.txt"],
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn rustc_unused_variable_as_hint() {
        check_with_config(
            DiagnosticsMapConfig {
                warnings_as_hint: vec!["unused_variables".to_owned()],
                ..DiagnosticsMapConfig::default()
            },
            r##"{
    "message": "unused variable: `foo`",
    "code": {
        "code": "unused_variables",
        "explanation": null
    },
    "level": "warning",
    "spans": [
        {
            "file_name": "driver/subcommand/repl.rs",
            "byte_start": 9228,
            "byte_end": 9231,
            "line_start": 291,
            "line_end": 291,
            "column_start": 9,
            "column_end": 12,
            "is_primary": true,
            "text": [
                {
                    "text": "    let foo = 42;",
                    "highlight_start": 9,
                    "highlight_end": 12
                }
            ],
            "label": null,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [
        {
            "message": "#[warn(unused_variables)] on by default",
            "code": null,
            "level": "note",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "consider prefixing with an underscore",
            "code": null,
            "level": "help",
            "spans": [
                {
                    "file_name": "driver/subcommand/repl.rs",
                    "byte_start": 9228,
                    "byte_end": 9231,
                    "line_start": 291,
                    "line_end": 291,
                    "column_start": 9,
                    "column_end": 12,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    let foo = 42;",
                            "highlight_start": 9,
                            "highlight_end": 12
                        }
                    ],
                    "label": null,
                    "suggested_replacement": "_foo",
                    "suggestion_applicability": "MachineApplicable",
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        }
    ],
    "rendered": "warning: unused variable: `foo`\n   --> driver/subcommand/repl.rs:291:9\n    |\n291 |     let foo = 42;\n    |         ^^^ help: consider prefixing with an underscore: `_foo`\n    |\n    = note: #[warn(unused_variables)] on by default\n\n"
    }"##,
            expect_file!["./test_data/rustc_unused_variable_as_hint.txt"],
        );
    }

    #[test]
    fn rustc_wrong_number_of_parameters() {
        check(
            r##"{
    "message": "this function takes 2 parameters but 3 parameters were supplied",
    "code": {
        "code": "E0061",
        "explanation": "\nThe number of arguments passed to a function must match the number of arguments\nspecified in the function signature.\n\nFor example, a function like:\n\n```\nfn f(a: u16, b: &str) {}\n```\n\nMust always be called with exactly two arguments, e.g., `f(2, \"test\")`.\n\nNote that Rust does not have a notion of optional function arguments or\nvariadic functions (except for its C-FFI).\n"
    },
    "level": "error",
    "spans": [
        {
            "file_name": "compiler/ty/select.rs",
            "byte_start": 8787,
            "byte_end": 9241,
            "line_start": 219,
            "line_end": 231,
            "column_start": 5,
            "column_end": 6,
            "is_primary": false,
            "text": [
                {
                    "text": "    pub fn add_evidence(",
                    "highlight_start": 5,
                    "highlight_end": 25
                },
                {
                    "text": "        &mut self,",
                    "highlight_start": 1,
                    "highlight_end": 19
                },
                {
                    "text": "        target_poly: &ty::Ref<ty::Poly>,",
                    "highlight_start": 1,
                    "highlight_end": 41
                },
                {
                    "text": "        evidence_poly: &ty::Ref<ty::Poly>,",
                    "highlight_start": 1,
                    "highlight_end": 43
                },
                {
                    "text": "    ) {",
                    "highlight_start": 1,
                    "highlight_end": 8
                },
                {
                    "text": "        match target_poly {",
                    "highlight_start": 1,
                    "highlight_end": 28
                },
                {
                    "text": "            ty::Ref::Var(tvar, _) => self.add_var_evidence(tvar, evidence_poly),",
                    "highlight_start": 1,
                    "highlight_end": 81
                },
                {
                    "text": "            ty::Ref::Fixed(target_ty) => {",
                    "highlight_start": 1,
                    "highlight_end": 43
                },
                {
                    "text": "                let evidence_ty = evidence_poly.resolve_to_ty();",
                    "highlight_start": 1,
                    "highlight_end": 65
                },
                {
                    "text": "                self.add_evidence_ty(target_ty, evidence_poly, evidence_ty)",
                    "highlight_start": 1,
                    "highlight_end": 76
                },
                {
                    "text": "            }",
                    "highlight_start": 1,
                    "highlight_end": 14
                },
                {
                    "text": "        }",
                    "highlight_start": 1,
                    "highlight_end": 10
                },
                {
                    "text": "    }",
                    "highlight_start": 1,
                    "highlight_end": 6
                }
            ],
            "label": "defined here",
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        },
        {
            "file_name": "compiler/ty/select.rs",
            "byte_start": 4045,
            "byte_end": 4057,
            "line_start": 104,
            "line_end": 104,
            "column_start": 18,
            "column_end": 30,
            "is_primary": true,
            "text": [
                {
                    "text": "            self.add_evidence(target_fixed, evidence_fixed, false);",
                    "highlight_start": 18,
                    "highlight_end": 30
                }
            ],
            "label": "expected 2 parameters",
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [],
    "rendered": "error[E0061]: this function takes 2 parameters but 3 parameters were supplied\n   --> compiler/ty/select.rs:104:18\n    |\n104 |               self.add_evidence(target_fixed, evidence_fixed, false);\n    |                    ^^^^^^^^^^^^ expected 2 parameters\n...\n219 | /     pub fn add_evidence(\n220 | |         &mut self,\n221 | |         target_poly: &ty::Ref<ty::Poly>,\n222 | |         evidence_poly: &ty::Ref<ty::Poly>,\n...   |\n230 | |         }\n231 | |     }\n    | |_____- defined here\n\n"
    }"##,
            expect_file!["./test_data/rustc_wrong_number_of_parameters.txt"],
        );
    }

    #[test]
    fn clippy_pass_by_ref() {
        check(
            r##"{
    "message": "this argument is passed by reference, but would be more efficient if passed by value",
    "code": {
        "code": "clippy::trivially_copy_pass_by_ref",
        "explanation": null
    },
    "level": "warning",
    "spans": [
        {
            "file_name": "compiler/mir/tagset.rs",
            "byte_start": 941,
            "byte_end": 946,
            "line_start": 42,
            "line_end": 42,
            "column_start": 24,
            "column_end": 29,
            "is_primary": true,
            "text": [
                {
                    "text": "    pub fn is_disjoint(&self, other: Self) -> bool {",
                    "highlight_start": 24,
                    "highlight_end": 29
                }
            ],
            "label": null,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [
        {
            "message": "lint level defined here",
            "code": null,
            "level": "note",
            "spans": [
                {
                    "file_name": "compiler/lib.rs",
                    "byte_start": 8,
                    "byte_end": 19,
                    "line_start": 1,
                    "line_end": 1,
                    "column_start": 9,
                    "column_end": 20,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "#![warn(clippy::all)]",
                            "highlight_start": 9,
                            "highlight_end": 20
                        }
                    ],
                    "label": null,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        },
        {
            "message": "#[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]",
            "code": null,
            "level": "note",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref",
            "code": null,
            "level": "help",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "consider passing by value instead",
            "code": null,
            "level": "help",
            "spans": [
                {
                    "file_name": "compiler/mir/tagset.rs",
                    "byte_start": 941,
                    "byte_end": 946,
                    "line_start": 42,
                    "line_end": 42,
                    "column_start": 24,
                    "column_end": 29,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    pub fn is_disjoint(&self, other: Self) -> bool {",
                            "highlight_start": 24,
                            "highlight_end": 29
                        }
                    ],
                    "label": null,
                    "suggested_replacement": "self",
                    "suggestion_applicability": "Unspecified",
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        }
    ],
    "rendered": "warning: this argument is passed by reference, but would be more efficient if passed by value\n  --> compiler/mir/tagset.rs:42:24\n   |\n42 |     pub fn is_disjoint(&self, other: Self) -> bool {\n   |                        ^^^^^ help: consider passing by value instead: `self`\n   |\nnote: lint level defined here\n  --> compiler/lib.rs:1:9\n   |\n1  | #![warn(clippy::all)]\n   |         ^^^^^^^^^^^\n   = note: #[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]\n   = help: for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref\n\n"
    }"##,
            expect_file!["./test_data/clippy_pass_by_ref.txt"],
        );
    }

    #[test]
    fn rustc_range_map_lsp_position() {
        check(
            r##"{
            "message": "mismatched types",
            "code": {
                "code": "E0308",
                "explanation": "Expected type did not match the received type.\n\nErroneous code examples:\n\n```compile_fail,E0308\nfn plus_one(x: i32) -> i32 {\n    x + 1\n}\n\nplus_one(\"Not a number\");\n//       ^^^^^^^^^^^^^^ expected `i32`, found `&str`\n\nif \"Not a bool\" {\n// ^^^^^^^^^^^^ expected `bool`, found `&str`\n}\n\nlet x: f32 = \"Not a float\";\n//     ---   ^^^^^^^^^^^^^ expected `f32`, found `&str`\n//     |\n//     expected due to this\n```\n\nThis error occurs when an expression was used in a place where the compiler\nexpected an expression of a different type. It can occur in several cases, the\nmost common being when calling a function and passing an argument which has a\ndifferent type than the matching type in the function declaration.\n"
            },
            "level": "error",
            "spans": [
                {
                    "file_name": "crates/test_diagnostics/src/main.rs",
                    "byte_start": 87,
                    "byte_end": 105,
                    "line_start": 4,
                    "line_end": 4,
                    "column_start": 18,
                    "column_end": 24,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    let x: u32 = \"𐐀𐐀𐐀𐐀\"; // 17-23",
                            "highlight_start": 18,
                            "highlight_end": 24
                        }
                    ],
                    "label": "expected `u32`, found `&str`",
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "expansion": null
                },
                {
                    "file_name": "crates/test_diagnostics/src/main.rs",
                    "byte_start": 81,
                    "byte_end": 84,
                    "line_start": 4,
                    "line_end": 4,
                    "column_start": 12,
                    "column_end": 15,
                    "is_primary": false,
                    "text": [
                        {
                            "text": "    let x: u32 = \"𐐀𐐀𐐀𐐀\"; // 17-23",
                            "highlight_start": 12,
                            "highlight_end": 15
                        }
                    ],
                    "label": "expected due to this",
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": "error[E0308]: mismatched types\n --> crates/test_diagnostics/src/main.rs:4:18\n  |\n4 |     let x: u32 = \"𐐀𐐀𐐀𐐀\"; // 17-23\n  |            ---   ^^^^^^ expected `u32`, found `&str`\n  |            |\n  |            expected due to this\n\n"
        }"##,
            expect_file!("./test_data/rustc_range_map_lsp_position.txt"),
        )
    }

    #[test]
    fn rustc_mismatched_type() {
        check(
            r##"{
    "message": "mismatched types",
    "code": {
        "code": "E0308",
        "explanation": "\nThis error occurs when the compiler was unable to infer the concrete type of a\nvariable. It can occur for several cases, the most common of which is a\nmismatch in the expected type that the compiler inferred for a variable's\ninitializing expression, and the actual type explicitly assigned to the\nvariable.\n\nFor example:\n\n```compile_fail,E0308\nlet x: i32 = \"I am not a number!\";\n//     ~~~   ~~~~~~~~~~~~~~~~~~~~\n//      |             |\n//      |    initializing expression;\n//      |    compiler infers type `&str`\n//      |\n//    type `i32` assigned to variable `x`\n```\n"
    },
    "level": "error",
    "spans": [
        {
            "file_name": "runtime/compiler_support.rs",
            "byte_start": 1589,
            "byte_end": 1594,
            "line_start": 48,
            "line_end": 48,
            "column_start": 65,
            "column_end": 70,
            "is_primary": true,
            "text": [
                {
                    "text": "    let layout = alloc::Layout::from_size_align_unchecked(size, align);",
                    "highlight_start": 65,
                    "highlight_end": 70
                }
            ],
            "label": "expected usize, found u32",
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [],
    "rendered": "error[E0308]: mismatched types\n  --> runtime/compiler_support.rs:48:65\n   |\n48 |     let layout = alloc::Layout::from_size_align_unchecked(size, align);\n   |                                                                 ^^^^^ expected usize, found u32\n\n"
    }"##,
            expect_file!["./test_data/rustc_mismatched_type.txt"],
        );
    }

    #[test]
    fn handles_macro_location() {
        check(
            r##"{
    "rendered": "error[E0277]: can't compare `{integer}` with `&str`\n --> src/main.rs:2:5\n  |\n2 |     assert_eq!(1, \"love\");\n  |     ^^^^^^^^^^^^^^^^^^^^^^ no implementation for `{integer} == &str`\n  |\n  = help: the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`\n  = note: this error originates in a macro outside of the current crate (in Nightly builds, run with -Z external-macro-backtrace for more info)\n\n",
    "children": [
        {
            "children": [],
            "code": null,
            "level": "help",
            "message": "the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`",
            "rendered": null,
            "spans": []
        }
    ],
    "code": {
        "code": "E0277",
        "explanation": "\nYou tried to use a type which doesn't implement some trait in a place which\nexpected that trait. Erroneous code example:\n\n```compile_fail,E0277\n// here we declare the Foo trait with a bar method\ntrait Foo {\n    fn bar(&self);\n}\n\n// we now declare a function which takes an object implementing the Foo trait\nfn some_func<T: Foo>(foo: T) {\n    foo.bar();\n}\n\nfn main() {\n    // we now call the method with the i32 type, which doesn't implement\n    // the Foo trait\n    some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied\n}\n```\n\nIn order to fix this error, verify that the type you're using does implement\nthe trait. Example:\n\n```\ntrait Foo {\n    fn bar(&self);\n}\n\nfn some_func<T: Foo>(foo: T) {\n    foo.bar(); // we can now use this method since i32 implements the\n               // Foo trait\n}\n\n// we implement the trait on the i32 type\nimpl Foo for i32 {\n    fn bar(&self) {}\n}\n\nfn main() {\n    some_func(5i32); // ok!\n}\n```\n\nOr in a generic context, an erroneous code example would look like:\n\n```compile_fail,E0277\nfn some_func<T>(foo: T) {\n    println!(\"{:?}\", foo); // error: the trait `core::fmt::Debug` is not\n                           //        implemented for the type `T`\n}\n\nfn main() {\n    // We now call the method with the i32 type,\n    // which *does* implement the Debug trait.\n    some_func(5i32);\n}\n```\n\nNote that the error here is in the definition of the generic function: Although\nwe only call it with a parameter that does implement `Debug`, the compiler\nstill rejects the function: It must work with all possible input types. In\norder to make this example compile, we need to restrict the generic type we're\naccepting:\n\n```\nuse std::fmt;\n\n// Restrict the input type to types that implement Debug.\nfn some_func<T: fmt::Debug>(foo: T) {\n    println!(\"{:?}\", foo);\n}\n\nfn main() {\n    // Calling the method is still fine, as i32 implements Debug.\n    some_func(5i32);\n\n    // This would fail to compile now:\n    // struct WithoutDebug;\n    // some_func(WithoutDebug);\n}\n```\n\nRust only looks at the signature of the called function, as such it must\nalready specify all requirements that will be used for every type parameter.\n"
    },
    "level": "error",
    "message": "can't compare `{integer}` with `&str`",
    "spans": [
        {
            "byte_end": 155,
            "byte_start": 153,
            "column_end": 33,
            "column_start": 31,
            "expansion": {
                "def_site_span": {
                    "byte_end": 940,
                    "byte_start": 0,
                    "column_end": 6,
                    "column_start": 1,
                    "expansion": null,
                    "file_name": "<::core::macros::assert_eq macros>",
                    "is_primary": false,
                    "label": null,
                    "line_end": 36,
                    "line_start": 1,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                            "highlight_end": 35,
                            "highlight_start": 1,
                            "text": "($ left : expr, $ right : expr) =>"
                        },
                        {
                            "highlight_end": 3,
                            "highlight_start": 1,
                            "text": "({"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "     match (& $ left, & $ right)"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     {"
                        },
                        {
                            "highlight_end": 34,
                            "highlight_start": 1,
                            "text": "         (left_val, right_val) =>"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         {"
                        },
                        {
                            "highlight_end": 46,
                            "highlight_start": 1,
                            "text": "             if ! (* left_val == * right_val)"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             {"
                        },
                        {
                            "highlight_end": 25,
                            "highlight_start": 1,
                            "text": "                 panic !"
                        },
                        {
                            "highlight_end": 57,
                            "highlight_start": 1,
                            "text": "                 (r#\"assertion failed: `(left == right)`"
                        },
                        {
                            "highlight_end": 16,
                            "highlight_start": 1,
                            "text": "  left: `{:?}`,"
                        },
                        {
                            "highlight_end": 18,
                            "highlight_start": 1,
                            "text": " right: `{:?}`\"#,"
                        },
                        {
                            "highlight_end": 47,
                            "highlight_start": 1,
                            "text": "                  & * left_val, & * right_val)"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             }"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         }"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     }"
                        },
                        {
                            "highlight_end": 42,
                            "highlight_start": 1,
                            "text": " }) ; ($ left : expr, $ right : expr,) =>"
                        },
                        {
                            "highlight_end": 49,
                            "highlight_start": 1,
                            "text": "({ $ crate :: assert_eq ! ($ left, $ right) }) ;"
                        },
                        {
                            "highlight_end": 53,
                            "highlight_start": 1,
                            "text": "($ left : expr, $ right : expr, $ ($ arg : tt) +) =>"
                        },
                        {
                            "highlight_end": 3,
                            "highlight_start": 1,
                            "text": "({"
                        },
                        {
                            "highlight_end": 37,
                            "highlight_start": 1,
                            "text": "     match (& ($ left), & ($ right))"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     {"
                        },
                        {
                            "highlight_end": 34,
                            "highlight_start": 1,
                            "text": "         (left_val, right_val) =>"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         {"
                        },
                        {
                            "highlight_end": 46,
                            "highlight_start": 1,
                            "text": "             if ! (* left_val == * right_val)"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             {"
                        },
                        {
                            "highlight_end": 25,
                            "highlight_start": 1,
                            "text": "                 panic !"
                        },
                        {
                            "highlight_end": 57,
                            "highlight_start": 1,
                            "text": "                 (r#\"assertion failed: `(left == right)`"
                        },
                        {
                            "highlight_end": 16,
                            "highlight_start": 1,
                            "text": "  left: `{:?}`,"
                        },
                        {
                            "highlight_end": 22,
                            "highlight_start": 1,
                            "text": " right: `{:?}`: {}\"#,"
                        },
                        {
                            "highlight_end": 72,
                            "highlight_start": 1,
                            "text": "                  & * left_val, & * right_val, $ crate :: format_args !"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "                  ($ ($ arg) +))"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             }"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         }"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     }"
                        },
                        {
                            "highlight_end": 6,
                            "highlight_start": 1,
                            "text": " }) ;"
                        }
                    ]
                },
                "macro_decl_name": "assert_eq!",
                "span": {
                    "byte_end": 38,
                    "byte_start": 16,
                    "column_end": 27,
                    "column_start": 5,
                    "expansion": null,
                    "file_name": "src/main.rs",
                    "is_primary": false,
                    "label": null,
                    "line_end": 2,
                    "line_start": 2,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                            "highlight_end": 27,
                            "highlight_start": 5,
                            "text": "    assert_eq!(1, \"love\");"
                        }
                    ]
                }
            },
            "file_name": "<::core::macros::assert_eq macros>",
            "is_primary": true,
            "label": "no implementation for `{integer} == &str`",
            "line_end": 7,
            "line_start": 7,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "text": [
                {
                    "highlight_end": 33,
                    "highlight_start": 31,
                    "text": "             if ! (* left_val == * right_val)"
                }
            ]
        }
    ]
    }"##,
            expect_file!["./test_data/handles_macro_location.txt"],
        );
    }

    #[test]
    fn macro_compiler_error() {
        check(
            r##"{
        "rendered": "error: Please register your known path in the path module\n   --> crates/hir_def/src/path.rs:265:9\n    |\n265 |         compile_error!(\"Please register your known path in the path module\")\n    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    | \n   ::: crates/hir_def/src/data.rs:80:16\n    |\n80  |     let path = path![std::future::Future];\n    |                -------------------------- in this macro invocation\n\n",
        "children": [],
        "code": null,
        "level": "error",
        "message": "Please register your known path in the path module",
        "spans": [
            {
                "byte_end": 8285,
                "byte_start": 8217,
                "column_end": 77,
                "column_start": 9,
                "expansion": {
                    "def_site_span": {
                        "byte_end": 8294,
                        "byte_start": 7858,
                        "column_end": 2,
                        "column_start": 1,
                        "expansion": null,
                        "file_name": "crates/hir_def/src/path.rs",
                        "is_primary": false,
                        "label": null,
                        "line_end": 267,
                        "line_start": 254,
                        "suggested_replacement": null,
                        "suggestion_applicability": null,
                        "text": [
                            {
                                "highlight_end": 28,
                                "highlight_start": 1,
                                "text": "macro_rules! __known_path {"
                            },
                            {
                                "highlight_end": 37,
                                "highlight_start": 1,
                                "text": "    (std::iter::IntoIterator) => {};"
                            },
                            {
                                "highlight_end": 33,
                                "highlight_start": 1,
                                "text": "    (std::result::Result) => {};"
                            },
                            {
                                "highlight_end": 29,
                                "highlight_start": 1,
                                "text": "    (std::ops::Range) => {};"
                            },
                            {
                                "highlight_end": 33,
                                "highlight_start": 1,
                                "text": "    (std::ops::RangeFrom) => {};"
                            },
                            {
                                "highlight_end": 33,
                                "highlight_start": 1,
                                "text": "    (std::ops::RangeFull) => {};"
                            },
                            {
                                "highlight_end": 31,
                                "highlight_start": 1,
                                "text": "    (std::ops::RangeTo) => {};"
                            },
                            {
                                "highlight_end": 40,
                                "highlight_start": 1,
                                "text": "    (std::ops::RangeToInclusive) => {};"
                            },
                            {
                                "highlight_end": 38,
                                "highlight_start": 1,
                                "text": "    (std::ops::RangeInclusive) => {};"
                            },
                            {
                                "highlight_end": 27,
                                "highlight_start": 1,
                                "text": "    (std::ops::Try) => {};"
                            },
                            {
                                "highlight_end": 22,
                                "highlight_start": 1,
                                "text": "    ($path:path) => {"
                            },
                            {
                                "highlight_end": 77,
                                "highlight_start": 1,
                                "text": "        compile_error!(\"Please register your known path in the path module\")"
                            },
                            {
                                "highlight_end": 7,
                                "highlight_start": 1,
                                "text": "    };"
                            },
                            {
                                "highlight_end": 2,
                                "highlight_start": 1,
                                "text": "}"
                            }
                        ]
                    },
                    "macro_decl_name": "$crate::__known_path!",
                    "span": {
                        "byte_end": 8427,
                        "byte_start": 8385,
                        "column_end": 51,
                        "column_start": 9,
                        "expansion": {
                            "def_site_span": {
                                "byte_end": 8611,
                                "byte_start": 8312,
                                "column_end": 2,
                                "column_start": 1,
                                "expansion": null,
                                "file_name": "crates/hir_def/src/path.rs",
                                "is_primary": false,
                                "label": null,
                                "line_end": 277,
                                "line_start": 270,
                                "suggested_replacement": null,
                                "suggestion_applicability": null,
                                "text": [
                                    {
                                        "highlight_end": 22,
                                        "highlight_start": 1,
                                        "text": "macro_rules! __path {"
                                    },
                                    {
                                        "highlight_end": 43,
                                        "highlight_start": 1,
                                        "text": "    ($start:ident $(:: $seg:ident)*) => ({"
                                    },
                                    {
                                        "highlight_end": 51,
                                        "highlight_start": 1,
                                        "text": "        $crate::__known_path!($start $(:: $seg)*);"
                                    },
                                    {
                                        "highlight_end": 87,
                                        "highlight_start": 1,
                                        "text": "        $crate::path::ModPath::from_simple_segments($crate::path::PathKind::Abs, vec!["
                                    },
                                    {
                                        "highlight_end": 76,
                                        "highlight_start": 1,
                                        "text": "            $crate::path::__name![$start], $($crate::path::__name![$seg],)*"
                                    },
                                    {
                                        "highlight_end": 11,
                                        "highlight_start": 1,
                                        "text": "        ])"
                                    },
                                    {
                                        "highlight_end": 8,
                                        "highlight_start": 1,
                                        "text": "    });"
                                    },
                                    {
                                        "highlight_end": 2,
                                        "highlight_start": 1,
                                        "text": "}"
                                    }
                                ]
                            },
                            "macro_decl_name": "path!",
                            "span": {
                                "byte_end": 2966,
                                "byte_start": 2940,
                                "column_end": 42,
                                "column_start": 16,
                                "expansion": null,
                                "file_name": "crates/hir_def/src/data.rs",
                                "is_primary": false,
                                "label": null,
                                "line_end": 80,
                                "line_start": 80,
                                "suggested_replacement": null,
                                "suggestion_applicability": null,
                                "text": [
                                    {
                                        "highlight_end": 42,
                                        "highlight_start": 16,
                                        "text": "    let path = path![std::future::Future];"
                                    }
                                ]
                            }
                        },
                        "file_name": "crates/hir_def/src/path.rs",
                        "is_primary": false,
                        "label": null,
                        "line_end": 272,
                        "line_start": 272,
                        "suggested_replacement": null,
                        "suggestion_applicability": null,
                        "text": [
                            {
                                "highlight_end": 51,
                                "highlight_start": 9,
                                "text": "        $crate::__known_path!($start $(:: $seg)*);"
                            }
                        ]
                    }
                },
                "file_name": "crates/hir_def/src/path.rs",
                "is_primary": true,
                "label": null,
                "line_end": 265,
                "line_start": 265,
                "suggested_replacement": null,
                "suggestion_applicability": null,
                "text": [
                    {
                        "highlight_end": 77,
                        "highlight_start": 9,
                        "text": "        compile_error!(\"Please register your known path in the path module\")"
                    }
                ]
            }
        ]
    }
            "##,
            expect_file!["./test_data/macro_compiler_error.txt"],
        );
    }

    #[test]
    fn snap_multi_line_fix() {
        check(
            r##"{
                "rendered": "warning: returning the result of a let binding from a block\n --> src/main.rs:4:5\n  |\n3 |     let a = (0..10).collect();\n  |     -------------------------- unnecessary let binding\n4 |     a\n  |     ^\n  |\n  = note: `#[warn(clippy::let_and_return)]` on by default\n  = help: for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#let_and_return\nhelp: return the expression directly\n  |\n3 |     \n4 |     (0..10).collect()\n  |\n\n",
                "children": [
                    {
                    "children": [],
                    "code": null,
                    "level": "note",
                    "message": "`#[warn(clippy::let_and_return)]` on by default",
                    "rendered": null,
                    "spans": []
                    },
                    {
                    "children": [],
                    "code": null,
                    "level": "help",
                    "message": "for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#let_and_return",
                    "rendered": null,
                    "spans": []
                    },
                    {
                    "children": [],
                    "code": null,
                    "level": "help",
                    "message": "return the expression directly",
                    "rendered": null,
                    "spans": [
                        {
                        "byte_end": 55,
                        "byte_start": 29,
                        "column_end": 31,
                        "column_start": 5,
                        "expansion": null,
                        "file_name": "src/main.rs",
                        "is_primary": true,
                        "label": null,
                        "line_end": 3,
                        "line_start": 3,
                        "suggested_replacement": "",
                        "suggestion_applicability": "MachineApplicable",
                        "text": [
                            {
                            "highlight_end": 31,
                            "highlight_start": 5,
                            "text": "    let a = (0..10).collect();"
                            }
                        ]
                        },
                        {
                        "byte_end": 61,
                        "byte_start": 60,
                        "column_end": 6,
                        "column_start": 5,
                        "expansion": null,
                        "file_name": "src/main.rs",
                        "is_primary": true,
                        "label": null,
                        "line_end": 4,
                        "line_start": 4,
                        "suggested_replacement": "(0..10).collect()",
                        "suggestion_applicability": "MachineApplicable",
                        "text": [
                            {
                            "highlight_end": 6,
                            "highlight_start": 5,
                            "text": "    a"
                            }
                        ]
                        }
                    ]
                    }
                ],
                "code": {
                    "code": "clippy::let_and_return",
                    "explanation": null
                },
                "level": "warning",
                "message": "returning the result of a let binding from a block",
                "spans": [
                    {
                    "byte_end": 55,
                    "byte_start": 29,
                    "column_end": 31,
                    "column_start": 5,
                    "expansion": null,
                    "file_name": "src/main.rs",
                    "is_primary": false,
                    "label": "unnecessary let binding",
                    "line_end": 3,
                    "line_start": 3,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                        "highlight_end": 31,
                        "highlight_start": 5,
                        "text": "    let a = (0..10).collect();"
                        }
                    ]
                    },
                    {
                    "byte_end": 61,
                    "byte_start": 60,
                    "column_end": 6,
                    "column_start": 5,
                    "expansion": null,
                    "file_name": "src/main.rs",
                    "is_primary": true,
                    "label": null,
                    "line_end": 4,
                    "line_start": 4,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                        "highlight_end": 6,
                        "highlight_start": 5,
                        "text": "    a"
                        }
                    ]
                    }
                ]
            }
            "##,
            expect_file!["./test_data/snap_multi_line_fix.txt"],
        );
    }

    #[test]
    fn reasonable_line_numbers_from_empty_file() {
        check(
            r##"{
                "message": "`main` function not found in crate `current`",
                "code": {
                    "code": "E0601",
                    "explanation": "No `main` function was found in a binary crate.\n\nTo fix this error, add a `main` function:\n\n```\nfn main() {\n    // Your program will start here.\n    println!(\"Hello world!\");\n}\n```\n\nIf you don't know the basics of Rust, you can look at the\n[Rust Book][rust-book] to get started.\n\n[rust-book]: https://doc.rust-lang.org/book/\n"
                },
                "level": "error",
                "spans": [
                    {
                        "file_name": "src/bin/current.rs",
                        "byte_start": 0,
                        "byte_end": 0,
                        "line_start": 0,
                        "line_end": 0,
                        "column_start": 1,
                        "column_end": 1,
                        "is_primary": true,
                        "text": [],
                        "label": null,
                        "suggested_replacement": null,
                        "suggestion_applicability": null,
                        "expansion": null
                    }
                ],
                "children": [
                    {
                        "message": "consider adding a `main` function to `src/bin/current.rs`",
                        "code": null,
                        "level": "note",
                        "spans": [],
                        "children": [],
                        "rendered": null
                    }
                ],
                "rendered": "error[E0601]: `main` function not found in crate `current`\n  |\n  = note: consider adding a `main` function to `src/bin/current.rs`\n\n"
            }"##,
            expect_file!["./test_data/reasonable_line_numbers_from_empty_file.txt"],
        );
    }
}
