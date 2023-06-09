//! A JSON emitter for errors.
//!
//! This works by converting errors to a simplified structural format (see the
//! structs at the start of the file) and then serializing them. These should
//! contain as much information about the error as possible.
//!
//! The format of the JSON output should be considered *unstable*. For now the
//! structs at the end of this file (Diagnostic*) specify the error format.

// FIXME: spec the JSON output properly.

use rustc_span::source_map::{FilePathMapping, SourceMap};

use crate::emitter::{Emitter, HumanReadableErrorType};
use crate::registry::Registry;
use crate::translation::{to_fluent_args, Translate};
use crate::DiagnosticId;
use crate::{
    CodeSuggestion, FluentBundle, LazyFallbackBundle, MultiSpan, SpanLabel, SubDiagnostic,
    TerminalUrl,
};
use rustc_lint_defs::Applicability;

use rustc_data_structures::sync::Lrc;
use rustc_error_messages::FluentArgs;
use rustc_span::hygiene::ExpnData;
use rustc_span::Span;
use std::error::Report;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::vec;

use serde::Serialize;

#[cfg(test)]
mod tests;

pub struct JsonEmitter {
    dst: Box<dyn Write + Send>,
    registry: Option<Registry>,
    sm: Lrc<SourceMap>,
    fluent_bundle: Option<Lrc<FluentBundle>>,
    fallback_bundle: LazyFallbackBundle,
    pretty: bool,
    ui_testing: bool,
    json_rendered: HumanReadableErrorType,
    diagnostic_width: Option<usize>,
    macro_backtrace: bool,
    track_diagnostics: bool,
    terminal_url: TerminalUrl,
}

impl JsonEmitter {
    pub fn stderr(
        registry: Option<Registry>,
        source_map: Lrc<SourceMap>,
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        diagnostic_width: Option<usize>,
        macro_backtrace: bool,
        track_diagnostics: bool,
        terminal_url: TerminalUrl,
    ) -> JsonEmitter {
        JsonEmitter {
            dst: Box::new(io::BufWriter::new(io::stderr())),
            registry,
            sm: source_map,
            fluent_bundle,
            fallback_bundle,
            pretty,
            ui_testing: false,
            json_rendered,
            diagnostic_width,
            macro_backtrace,
            track_diagnostics,
            terminal_url,
        }
    }

    pub fn basic(
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        diagnostic_width: Option<usize>,
        macro_backtrace: bool,
        track_diagnostics: bool,
        terminal_url: TerminalUrl,
    ) -> JsonEmitter {
        let file_path_mapping = FilePathMapping::empty();
        JsonEmitter::stderr(
            None,
            Lrc::new(SourceMap::new(file_path_mapping)),
            fluent_bundle,
            fallback_bundle,
            pretty,
            json_rendered,
            diagnostic_width,
            macro_backtrace,
            track_diagnostics,
            terminal_url,
        )
    }

    pub fn new(
        dst: Box<dyn Write + Send>,
        registry: Option<Registry>,
        source_map: Lrc<SourceMap>,
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        diagnostic_width: Option<usize>,
        macro_backtrace: bool,
        track_diagnostics: bool,
        terminal_url: TerminalUrl,
    ) -> JsonEmitter {
        JsonEmitter {
            dst,
            registry,
            sm: source_map,
            fluent_bundle,
            fallback_bundle,
            pretty,
            ui_testing: false,
            json_rendered,
            diagnostic_width,
            macro_backtrace,
            track_diagnostics,
            terminal_url,
        }
    }

    pub fn ui_testing(self, ui_testing: bool) -> Self {
        Self { ui_testing, ..self }
    }
}

impl Translate for JsonEmitter {
    fn fluent_bundle(&self) -> Option<&Lrc<FluentBundle>> {
        self.fluent_bundle.as_ref()
    }

    fn fallback_fluent_bundle(&self) -> &FluentBundle {
        &self.fallback_bundle
    }
}

impl Emitter for JsonEmitter {
    fn emit_diagnostic(&mut self, diag: &crate::Diagnostic) {
        let data = Diagnostic::from_errors_diagnostic(diag, self);
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", serde_json::to_string_pretty(&data).unwrap())
        } else {
            writeln!(&mut self.dst, "{}", serde_json::to_string(&data).unwrap())
        }
        .and_then(|_| self.dst.flush());
        if let Err(e) = result {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }

    fn emit_artifact_notification(&mut self, path: &Path, artifact_type: &str) {
        let data = ArtifactNotification { artifact: path, emit: artifact_type };
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", serde_json::to_string_pretty(&data).unwrap())
        } else {
            writeln!(&mut self.dst, "{}", serde_json::to_string(&data).unwrap())
        }
        .and_then(|_| self.dst.flush());
        if let Err(e) = result {
            panic!("failed to print notification: {:?}", e);
        }
    }

    fn emit_future_breakage_report(&mut self, diags: Vec<crate::Diagnostic>) {
        let data: Vec<FutureBreakageItem> = diags
            .into_iter()
            .map(|mut diag| {
                if diag.level == crate::Level::Allow {
                    diag.level = crate::Level::Warning(None);
                }
                FutureBreakageItem { diagnostic: Diagnostic::from_errors_diagnostic(&diag, self) }
            })
            .collect();
        let report = FutureIncompatReport { future_incompat_report: data };
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", serde_json::to_string_pretty(&report).unwrap())
        } else {
            writeln!(&mut self.dst, "{}", serde_json::to_string(&report).unwrap())
        }
        .and_then(|_| self.dst.flush());
        if let Err(e) = result {
            panic!("failed to print future breakage report: {:?}", e);
        }
    }

    fn emit_unused_externs(&mut self, lint_level: rustc_lint_defs::Level, unused_externs: &[&str]) {
        let lint_level = lint_level.as_str();
        let data = UnusedExterns { lint_level, unused_extern_names: unused_externs };
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", serde_json::to_string_pretty(&data).unwrap())
        } else {
            writeln!(&mut self.dst, "{}", serde_json::to_string(&data).unwrap())
        }
        .and_then(|_| self.dst.flush());
        if let Err(e) = result {
            panic!("failed to print unused externs: {:?}", e);
        }
    }

    fn source_map(&self) -> Option<&Lrc<SourceMap>> {
        Some(&self.sm)
    }

    fn should_show_explain(&self) -> bool {
        !matches!(self.json_rendered, HumanReadableErrorType::Short(_))
    }
}

// The following data types are provided just for serialisation.

#[derive(Serialize)]
struct Diagnostic {
    /// The primary error message.
    message: String,
    code: Option<DiagnosticCode>,
    /// "error: internal compiler error", "error", "warning", "note", "help".
    level: &'static str,
    spans: Vec<DiagnosticSpan>,
    /// Associated diagnostic messages.
    children: Vec<Diagnostic>,
    /// The message as rustc would render it.
    rendered: Option<String>,
}

#[derive(Serialize)]
struct DiagnosticSpan {
    file_name: String,
    byte_start: u32,
    byte_end: u32,
    /// 1-based.
    line_start: usize,
    line_end: usize,
    /// 1-based, character offset.
    column_start: usize,
    column_end: usize,
    /// Is this a "primary" span -- meaning the point, or one of the points,
    /// where the error occurred?
    is_primary: bool,
    /// Source text from the start of line_start to the end of line_end.
    text: Vec<DiagnosticSpanLine>,
    /// Label that should be placed at this location (if any)
    label: Option<String>,
    /// If we are suggesting a replacement, this will contain text
    /// that should be sliced in atop this span.
    suggested_replacement: Option<String>,
    /// If the suggestion is approximate
    suggestion_applicability: Option<Applicability>,
    /// Macro invocations that created the code at this span, if any.
    expansion: Option<Box<DiagnosticSpanMacroExpansion>>,
}

#[derive(Serialize)]
struct DiagnosticSpanLine {
    text: String,

    /// 1-based, character offset in self.text.
    highlight_start: usize,

    highlight_end: usize,
}

#[derive(Serialize)]
struct DiagnosticSpanMacroExpansion {
    /// span where macro was applied to generate this code; note that
    /// this may itself derive from a macro (if
    /// `span.expansion.is_some()`)
    span: DiagnosticSpan,

    /// name of macro that was applied (e.g., "foo!" or "#[derive(Eq)]")
    macro_decl_name: String,

    /// span where macro was defined (if known)
    def_site_span: DiagnosticSpan,
}

#[derive(Serialize)]
struct DiagnosticCode {
    /// The code itself.
    code: String,
    /// An explanation for the code.
    explanation: Option<&'static str>,
}

#[derive(Serialize)]
struct ArtifactNotification<'a> {
    /// The path of the artifact.
    artifact: &'a Path,
    /// What kind of artifact we're emitting.
    emit: &'a str,
}

#[derive(Serialize)]
struct FutureBreakageItem {
    diagnostic: Diagnostic,
}

#[derive(Serialize)]
struct FutureIncompatReport {
    future_incompat_report: Vec<FutureBreakageItem>,
}

// NOTE: Keep this in sync with the equivalent structs in rustdoc's
// doctest component (as well as cargo).
// We could unify this struct the one in rustdoc but they have different
// ownership semantics, so doing so would create wasteful allocations.
#[derive(Serialize)]
struct UnusedExterns<'a, 'b, 'c> {
    /// The severity level of the unused dependencies lint
    lint_level: &'a str,
    /// List of unused externs by their names.
    unused_extern_names: &'b [&'c str],
}

impl Diagnostic {
    fn from_errors_diagnostic(diag: &crate::Diagnostic, je: &JsonEmitter) -> Diagnostic {
        let args = to_fluent_args(diag.args());
        let sugg = diag.suggestions.iter().flatten().map(|sugg| {
            let translated_message =
                je.translate_message(&sugg.msg, &args).map_err(Report::new).unwrap();
            Diagnostic {
                message: translated_message.to_string(),
                code: None,
                level: "help",
                spans: DiagnosticSpan::from_suggestion(sugg, &args, je),
                children: vec![],
                rendered: None,
            }
        });

        // generate regular command line output and store it in the json

        // A threadsafe buffer for writing.
        #[derive(Default, Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.0.lock().unwrap().flush()
            }
        }
        let buf = BufWriter::default();
        let output = buf.clone();
        je.json_rendered
            .new_emitter(
                Box::new(buf),
                Some(je.sm.clone()),
                je.fluent_bundle.clone(),
                je.fallback_bundle.clone(),
                false,
                je.diagnostic_width,
                je.macro_backtrace,
                je.track_diagnostics,
                je.terminal_url,
            )
            .ui_testing(je.ui_testing)
            .emit_diagnostic(diag);
        let output = Arc::try_unwrap(output.0).unwrap().into_inner().unwrap();
        let output = String::from_utf8(output).unwrap();

        let translated_message = je.translate_messages(&diag.message, &args);
        Diagnostic {
            message: translated_message.to_string(),
            code: DiagnosticCode::map_opt_string(diag.code.clone(), je),
            level: diag.level.to_str(),
            spans: DiagnosticSpan::from_multispan(&diag.span, &args, je),
            children: diag
                .children
                .iter()
                .map(|c| Diagnostic::from_sub_diagnostic(c, &args, je))
                .chain(sugg)
                .collect(),
            rendered: Some(output),
        }
    }

    fn from_sub_diagnostic(
        diag: &SubDiagnostic,
        args: &FluentArgs<'_>,
        je: &JsonEmitter,
    ) -> Diagnostic {
        let translated_message = je.translate_messages(&diag.message, args);
        Diagnostic {
            message: translated_message.to_string(),
            code: None,
            level: diag.level.to_str(),
            spans: diag
                .render_span
                .as_ref()
                .map(|sp| DiagnosticSpan::from_multispan(sp, args, je))
                .unwrap_or_else(|| DiagnosticSpan::from_multispan(&diag.span, args, je)),
            children: vec![],
            rendered: None,
        }
    }
}

impl DiagnosticSpan {
    fn from_span_label(
        span: SpanLabel,
        suggestion: Option<(&String, Applicability)>,
        args: &FluentArgs<'_>,
        je: &JsonEmitter,
    ) -> DiagnosticSpan {
        Self::from_span_etc(
            span.span,
            span.is_primary,
            span.label
                .as_ref()
                .map(|m| je.translate_message(m, args).unwrap())
                .map(|m| m.to_string()),
            suggestion,
            je,
        )
    }

    fn from_span_etc(
        span: Span,
        is_primary: bool,
        label: Option<String>,
        suggestion: Option<(&String, Applicability)>,
        je: &JsonEmitter,
    ) -> DiagnosticSpan {
        // obtain the full backtrace from the `macro_backtrace`
        // helper; in some ways, it'd be better to expand the
        // backtrace ourselves, but the `macro_backtrace` helper makes
        // some decision, such as dropping some frames, and I don't
        // want to duplicate that logic here.
        let backtrace = span.macro_backtrace();
        DiagnosticSpan::from_span_full(span, is_primary, label, suggestion, backtrace, je)
    }

    fn from_span_full(
        span: Span,
        is_primary: bool,
        label: Option<String>,
        suggestion: Option<(&String, Applicability)>,
        mut backtrace: impl Iterator<Item = ExpnData>,
        je: &JsonEmitter,
    ) -> DiagnosticSpan {
        let start = je.sm.lookup_char_pos(span.lo());
        let end = je.sm.lookup_char_pos(span.hi());
        let backtrace_step = backtrace.next().map(|bt| {
            let call_site = Self::from_span_full(bt.call_site, false, None, None, backtrace, je);
            let def_site_span = Self::from_span_full(
                je.sm.guess_head_span(bt.def_site),
                false,
                None,
                None,
                [].into_iter(),
                je,
            );
            Box::new(DiagnosticSpanMacroExpansion {
                span: call_site,
                macro_decl_name: bt.kind.descr(),
                def_site_span,
            })
        });

        DiagnosticSpan {
            file_name: je.sm.filename_for_diagnostics(&start.file.name).to_string(),
            byte_start: start.file.original_relative_byte_pos(span.lo()).0,
            byte_end: start.file.original_relative_byte_pos(span.hi()).0,
            line_start: start.line,
            line_end: end.line,
            column_start: start.col.0 + 1,
            column_end: end.col.0 + 1,
            is_primary,
            text: DiagnosticSpanLine::from_span(span, je),
            suggested_replacement: suggestion.map(|x| x.0.clone()),
            suggestion_applicability: suggestion.map(|x| x.1),
            expansion: backtrace_step,
            label,
        }
    }

    fn from_multispan(
        msp: &MultiSpan,
        args: &FluentArgs<'_>,
        je: &JsonEmitter,
    ) -> Vec<DiagnosticSpan> {
        msp.span_labels()
            .into_iter()
            .map(|span_str| Self::from_span_label(span_str, None, args, je))
            .collect()
    }

    fn from_suggestion(
        suggestion: &CodeSuggestion,
        args: &FluentArgs<'_>,
        je: &JsonEmitter,
    ) -> Vec<DiagnosticSpan> {
        suggestion
            .substitutions
            .iter()
            .flat_map(|substitution| {
                substitution.parts.iter().map(move |suggestion_inner| {
                    let span_label =
                        SpanLabel { span: suggestion_inner.span, is_primary: true, label: None };
                    DiagnosticSpan::from_span_label(
                        span_label,
                        Some((&suggestion_inner.snippet, suggestion.applicability)),
                        args,
                        je,
                    )
                })
            })
            .collect()
    }
}

impl DiagnosticSpanLine {
    fn line_from_source_file(
        sf: &rustc_span::SourceFile,
        index: usize,
        h_start: usize,
        h_end: usize,
    ) -> DiagnosticSpanLine {
        DiagnosticSpanLine {
            text: sf.get_line(index).map_or_else(String::new, |l| l.into_owned()),
            highlight_start: h_start,
            highlight_end: h_end,
        }
    }

    /// Creates a list of DiagnosticSpanLines from span - each line with any part
    /// of `span` gets a DiagnosticSpanLine, with the highlight indicating the
    /// `span` within the line.
    fn from_span(span: Span, je: &JsonEmitter) -> Vec<DiagnosticSpanLine> {
        je.sm
            .span_to_lines(span)
            .map(|lines| {
                // We can't get any lines if the source is unavailable.
                if !je.sm.ensure_source_file_source_present(lines.file.clone()) {
                    return vec![];
                }

                let sf = &*lines.file;
                lines
                    .lines
                    .iter()
                    .map(|line| {
                        DiagnosticSpanLine::line_from_source_file(
                            sf,
                            line.line_index,
                            line.start_col.0 + 1,
                            line.end_col.0 + 1,
                        )
                    })
                    .collect()
            })
            .unwrap_or_else(|_| vec![])
    }
}

impl DiagnosticCode {
    fn map_opt_string(s: Option<DiagnosticId>, je: &JsonEmitter) -> Option<DiagnosticCode> {
        s.map(|s| {
            let s = match s {
                DiagnosticId::Error(s) => s,
                DiagnosticId::Lint { name, .. } => name,
            };
            let je_result =
                je.registry.as_ref().map(|registry| registry.try_find_description(&s)).unwrap();

            DiagnosticCode { code: s, explanation: je_result.ok() }
        })
    }
}
