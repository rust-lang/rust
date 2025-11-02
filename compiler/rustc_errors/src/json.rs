//! A JSON emitter for errors.
//!
//! This works by converting errors to a simplified structural format (see the
//! structs at the start of the file) and then serializing them. These should
//! contain as much information about the error as possible.
//!
//! The format of the JSON output should be considered *unstable*. For now the
//! structs at the end of this file (Diagnostic*) specify the error format.

// FIXME: spec the JSON output properly.

use std::error::Report;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::vec;

use anstream::{AutoStream, ColorChoice};
use derive_setters::Setters;
use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_error_messages::FluentArgs;
use rustc_lint_defs::Applicability;
use rustc_span::Span;
use rustc_span::hygiene::ExpnData;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use serde::Serialize;

use crate::annotate_snippet_emitter_writer::AnnotateSnippetEmitter;
use crate::diagnostic::IsLint;
use crate::emitter::{
    ColorConfig, Destination, Emitter, HumanEmitter, HumanReadableErrorType, OutputTheme,
    TimingEvent, should_show_source_code,
};
use crate::registry::Registry;
use crate::timings::{TimingRecord, TimingSection};
use crate::translation::{Translator, to_fluent_args};
use crate::{CodeSuggestion, MultiSpan, SpanLabel, Subdiag, Suggestions, TerminalUrl};

#[cfg(test)]
mod tests;

#[derive(Setters)]
pub struct JsonEmitter {
    #[setters(skip)]
    dst: IntoDynSyncSend<Box<dyn Write + Send>>,
    #[setters(skip)]
    sm: Option<Arc<SourceMap>>,
    #[setters(skip)]
    translator: Translator,
    #[setters(skip)]
    pretty: bool,
    ui_testing: bool,
    ignored_directories_in_source_blocks: Vec<String>,
    #[setters(skip)]
    json_rendered: HumanReadableErrorType,
    color_config: ColorConfig,
    diagnostic_width: Option<usize>,
    macro_backtrace: bool,
    track_diagnostics: bool,
    terminal_url: TerminalUrl,
}

impl JsonEmitter {
    pub fn new(
        dst: Box<dyn Write + Send>,
        sm: Option<Arc<SourceMap>>,
        translator: Translator,
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        color_config: ColorConfig,
    ) -> JsonEmitter {
        JsonEmitter {
            dst: IntoDynSyncSend(dst),
            sm,
            translator,
            pretty,
            ui_testing: false,
            ignored_directories_in_source_blocks: Vec::new(),
            json_rendered,
            color_config,
            diagnostic_width: None,
            macro_backtrace: false,
            track_diagnostics: false,
            terminal_url: TerminalUrl::No,
        }
    }

    fn emit(&mut self, val: EmitTyped<'_>) -> io::Result<()> {
        if self.pretty {
            serde_json::to_writer_pretty(&mut *self.dst, &val)?
        } else {
            serde_json::to_writer(&mut *self.dst, &val)?
        };
        self.dst.write_all(b"\n")?;
        self.dst.flush()
    }
}

#[derive(Serialize)]
#[serde(tag = "$message_type", rename_all = "snake_case")]
enum EmitTyped<'a> {
    Diagnostic(Diagnostic),
    Artifact(ArtifactNotification<'a>),
    SectionTiming(SectionTimestamp<'a>),
    FutureIncompat(FutureIncompatReport<'a>),
    UnusedExtern(UnusedExterns<'a>),
}

impl Emitter for JsonEmitter {
    fn emit_diagnostic(&mut self, diag: crate::DiagInner, registry: &Registry) {
        let data = Diagnostic::from_errors_diagnostic(diag, self, registry);
        let result = self.emit(EmitTyped::Diagnostic(data));
        if let Err(e) = result {
            panic!("failed to print diagnostics: {e:?}");
        }
    }

    fn emit_artifact_notification(&mut self, path: &Path, artifact_type: &str) {
        let data = ArtifactNotification { artifact: path, emit: artifact_type };
        let result = self.emit(EmitTyped::Artifact(data));
        if let Err(e) = result {
            panic!("failed to print notification: {e:?}");
        }
    }

    fn emit_timing_section(&mut self, record: TimingRecord, event: TimingEvent) {
        let event = match event {
            TimingEvent::Start => "start",
            TimingEvent::End => "end",
        };
        let name = match record.section {
            TimingSection::Linking => "link",
            TimingSection::Codegen => "codegen",
        };
        let data = SectionTimestamp { name, event, timestamp: record.timestamp };
        let result = self.emit(EmitTyped::SectionTiming(data));
        if let Err(e) = result {
            panic!("failed to print timing section: {e:?}");
        }
    }

    fn emit_future_breakage_report(&mut self, diags: Vec<crate::DiagInner>, registry: &Registry) {
        let data: Vec<FutureBreakageItem<'_>> = diags
            .into_iter()
            .map(|mut diag| {
                // Allowed or expected lints don't normally (by definition) emit a lint
                // but future incompat lints are special and are emitted anyway.
                //
                // So to avoid ICEs and confused users we "upgrade" the lint level for
                // those `FutureBreakageItem` to warn.
                if matches!(diag.level, crate::Level::Allow | crate::Level::Expect) {
                    diag.level = crate::Level::Warning;
                }
                FutureBreakageItem {
                    diagnostic: EmitTyped::Diagnostic(Diagnostic::from_errors_diagnostic(
                        diag, self, registry,
                    )),
                }
            })
            .collect();
        let report = FutureIncompatReport { future_incompat_report: data };
        let result = self.emit(EmitTyped::FutureIncompat(report));
        if let Err(e) = result {
            panic!("failed to print future breakage report: {e:?}");
        }
    }

    fn emit_unused_externs(&mut self, lint_level: rustc_lint_defs::Level, unused_externs: &[&str]) {
        let lint_level = lint_level.as_str();
        let data = UnusedExterns { lint_level, unused_extern_names: unused_externs };
        let result = self.emit(EmitTyped::UnusedExtern(data));
        if let Err(e) = result {
            panic!("failed to print unused externs: {e:?}");
        }
    }

    fn source_map(&self) -> Option<&SourceMap> {
        self.sm.as_deref()
    }

    fn should_show_explain(&self) -> bool {
        !self.json_rendered.short()
    }

    fn translator(&self) -> &Translator {
        &self.translator
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
    /// The error code (e.g. "E1234"), if the diagnostic has one. Or the lint
    /// name, if it's a lint without an error code.
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
struct SectionTimestamp<'a> {
    /// Name of the section
    name: &'a str,
    /// Start/end of the section
    event: &'a str,
    /// Opaque timestamp.
    timestamp: u128,
}

#[derive(Serialize)]
struct FutureBreakageItem<'a> {
    // Always EmitTyped::Diagnostic, but we want to make sure it gets serialized
    // with "$message_type".
    diagnostic: EmitTyped<'a>,
}

#[derive(Serialize)]
struct FutureIncompatReport<'a> {
    future_incompat_report: Vec<FutureBreakageItem<'a>>,
}

// NOTE: Keep this in sync with the equivalent structs in rustdoc's
// doctest component (as well as cargo).
// We could unify this struct the one in rustdoc but they have different
// ownership semantics, so doing so would create wasteful allocations.
#[derive(Serialize)]
struct UnusedExterns<'a> {
    /// The severity level of the unused dependencies lint
    lint_level: &'a str,
    /// List of unused externs by their names.
    unused_extern_names: &'a [&'a str],
}

impl Diagnostic {
    /// Converts from `rustc_errors::DiagInner` to `Diagnostic`.
    fn from_errors_diagnostic(
        diag: crate::DiagInner,
        je: &JsonEmitter,
        registry: &Registry,
    ) -> Diagnostic {
        let args = to_fluent_args(diag.args.iter());
        let sugg_to_diag = |sugg: &CodeSuggestion| {
            let translated_message =
                je.translator.translate_message(&sugg.msg, &args).map_err(Report::new).unwrap();
            Diagnostic {
                message: translated_message.to_string(),
                code: None,
                level: "help",
                spans: DiagnosticSpan::from_suggestion(sugg, &args, je),
                children: vec![],
                rendered: None,
            }
        };
        let sugg = match &diag.suggestions {
            Suggestions::Enabled(suggestions) => suggestions.iter().map(sugg_to_diag),
            Suggestions::Sealed(suggestions) => suggestions.iter().map(sugg_to_diag),
            Suggestions::Disabled => [].iter().map(sugg_to_diag),
        };

        // generate regular command line output and store it in the json

        // A threadsafe buffer for writing.
        #[derive(Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.0.lock().unwrap().flush()
            }
        }

        let translated_message = je.translator.translate_messages(&diag.messages, &args);

        let code = if let Some(code) = diag.code {
            Some(DiagnosticCode {
                code: code.to_string(),
                explanation: registry.try_find_description(code).ok(),
            })
        } else if let Some(IsLint { name, .. }) = &diag.is_lint {
            Some(DiagnosticCode { code: name.to_string(), explanation: None })
        } else {
            None
        };
        let level = diag.level.to_str();
        let spans = DiagnosticSpan::from_multispan(&diag.span, &args, je);
        let mut children: Vec<Diagnostic> = diag
            .children
            .iter()
            .map(|c| Diagnostic::from_sub_diagnostic(c, &args, je))
            .chain(sugg)
            .collect();
        if je.track_diagnostics && diag.span.has_primary_spans() && !diag.span.is_dummy() {
            children
                .insert(0, Diagnostic::from_sub_diagnostic(&diag.emitted_at_sub_diag(), &args, je));
        }
        let buf = BufWriter(Arc::new(Mutex::new(Vec::new())));
        let dst: Destination = AutoStream::new(
            Box::new(buf.clone()),
            match je.color_config.to_color_choice() {
                ColorChoice::Auto => ColorChoice::Always,
                choice => choice,
            },
        );
        match je.json_rendered {
            HumanReadableErrorType::AnnotateSnippet { short, unicode } => {
                AnnotateSnippetEmitter::new(dst, je.translator.clone())
                    .short_message(short)
                    .sm(je.sm.clone())
                    .diagnostic_width(je.diagnostic_width)
                    .macro_backtrace(je.macro_backtrace)
                    .track_diagnostics(je.track_diagnostics)
                    .terminal_url(je.terminal_url)
                    .ui_testing(je.ui_testing)
                    .ignored_directories_in_source_blocks(
                        je.ignored_directories_in_source_blocks.clone(),
                    )
                    .theme(if unicode { OutputTheme::Unicode } else { OutputTheme::Ascii })
                    .emit_diagnostic(diag, registry)
            }
            HumanReadableErrorType::Default { short } => {
                HumanEmitter::new(dst, je.translator.clone())
                    .short_message(short)
                    .sm(je.sm.clone())
                    .diagnostic_width(je.diagnostic_width)
                    .macro_backtrace(je.macro_backtrace)
                    .track_diagnostics(je.track_diagnostics)
                    .terminal_url(je.terminal_url)
                    .ui_testing(je.ui_testing)
                    .ignored_directories_in_source_blocks(
                        je.ignored_directories_in_source_blocks.clone(),
                    )
                    .theme(OutputTheme::Ascii)
                    .emit_diagnostic(diag, registry)
            }
        }

        let buf = Arc::try_unwrap(buf.0).unwrap().into_inner().unwrap();
        let buf = String::from_utf8(buf).unwrap();

        Diagnostic {
            message: translated_message.to_string(),
            code,
            level,
            spans,
            children,
            rendered: Some(buf),
        }
    }

    fn from_sub_diagnostic(
        subdiag: &Subdiag,
        args: &FluentArgs<'_>,
        je: &JsonEmitter,
    ) -> Diagnostic {
        let translated_message = je.translator.translate_messages(&subdiag.messages, args);
        Diagnostic {
            message: translated_message.to_string(),
            code: None,
            level: subdiag.level.to_str(),
            spans: DiagnosticSpan::from_multispan(&subdiag.span, args, je),
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
                .map(|m| je.translator.translate_message(m, args).unwrap())
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
        mut span: Span,
        is_primary: bool,
        label: Option<String>,
        suggestion: Option<(&String, Applicability)>,
        mut backtrace: impl Iterator<Item = ExpnData>,
        je: &JsonEmitter,
    ) -> DiagnosticSpan {
        let empty_source_map;
        let sm = match &je.sm {
            Some(s) => s,
            None => {
                span = rustc_span::DUMMY_SP;
                empty_source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
                empty_source_map
                    .new_source_file(std::path::PathBuf::from("empty.rs").into(), String::new());
                &empty_source_map
            }
        };
        let start = sm.lookup_char_pos(span.lo());
        // If this goes from the start of a line to the end and the replacement
        // is an empty string, increase the length to include the newline so we don't
        // leave an empty line
        if start.col.0 == 0
            && let Some((suggestion, _)) = suggestion
            && suggestion.is_empty()
            && let Ok(after) = sm.span_to_next_source(span)
            && after.starts_with('\n')
        {
            span = span.with_hi(span.hi() + rustc_span::BytePos(1));
        }
        let end = sm.lookup_char_pos(span.hi());
        let backtrace_step = backtrace.next().map(|bt| {
            let call_site = Self::from_span_full(bt.call_site, false, None, None, backtrace, je);
            let def_site_span = Self::from_span_full(
                sm.guess_head_span(bt.def_site),
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
            file_name: sm.filename_for_diagnostics(&start.file.name).to_string(),
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
            .as_ref()
            .and_then(|sm| {
                let lines = sm.span_to_lines(span).ok()?;
                // We can't get any lines if the source is unavailable.
                if !should_show_source_code(
                    &je.ignored_directories_in_source_blocks,
                    &sm,
                    &lines.file,
                ) {
                    return None;
                }

                let sf = &*lines.file;
                let span_lines = lines
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
                    .collect();
                Some(span_lines)
            })
            .unwrap_or_default()
    }
}
