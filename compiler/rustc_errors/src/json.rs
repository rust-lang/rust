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
use crate::DiagnosticId;
use crate::ToolMetadata;
use crate::{CodeSuggestion, SubDiagnostic};
use rustc_lint_defs::Applicability;

use rustc_data_structures::sync::Lrc;
use rustc_span::hygiene::ExpnData;
use rustc_span::{MultiSpan, Span, SpanLabel};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::vec;

use rustc_serialize::json::{as_json, as_pretty_json};
use rustc_serialize::{Encodable, Encoder};

#[cfg(test)]
mod tests;

pub struct JsonEmitter {
    dst: Box<dyn Write + Send>,
    registry: Option<Registry>,
    sm: Lrc<SourceMap>,
    pretty: bool,
    ui_testing: bool,
    json_rendered: HumanReadableErrorType,
    terminal_width: Option<usize>,
    macro_backtrace: bool,
}

impl JsonEmitter {
    pub fn stderr(
        registry: Option<Registry>,
        source_map: Lrc<SourceMap>,
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        terminal_width: Option<usize>,
        macro_backtrace: bool,
    ) -> JsonEmitter {
        JsonEmitter {
            dst: Box::new(io::BufWriter::new(io::stderr())),
            registry,
            sm: source_map,
            pretty,
            ui_testing: false,
            json_rendered,
            terminal_width,
            macro_backtrace,
        }
    }

    pub fn basic(
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        terminal_width: Option<usize>,
        macro_backtrace: bool,
    ) -> JsonEmitter {
        let file_path_mapping = FilePathMapping::empty();
        JsonEmitter::stderr(
            None,
            Lrc::new(SourceMap::new(file_path_mapping)),
            pretty,
            json_rendered,
            terminal_width,
            macro_backtrace,
        )
    }

    pub fn new(
        dst: Box<dyn Write + Send>,
        registry: Option<Registry>,
        source_map: Lrc<SourceMap>,
        pretty: bool,
        json_rendered: HumanReadableErrorType,
        terminal_width: Option<usize>,
        macro_backtrace: bool,
    ) -> JsonEmitter {
        JsonEmitter {
            dst,
            registry,
            sm: source_map,
            pretty,
            ui_testing: false,
            json_rendered,
            terminal_width,
            macro_backtrace,
        }
    }

    pub fn ui_testing(self, ui_testing: bool) -> Self {
        Self { ui_testing, ..self }
    }
}

impl Emitter for JsonEmitter {
    fn emit_diagnostic(&mut self, diag: &crate::Diagnostic) {
        let data = Diagnostic::from_errors_diagnostic(diag, self);
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", as_pretty_json(&data))
        } else {
            writeln!(&mut self.dst, "{}", as_json(&data))
        }
        .and_then(|_| self.dst.flush());
        if let Err(e) = result {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }

    fn emit_artifact_notification(&mut self, path: &Path, artifact_type: &str) {
        let data = ArtifactNotification { artifact: path, emit: artifact_type };
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", as_pretty_json(&data))
        } else {
            writeln!(&mut self.dst, "{}", as_json(&data))
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
                    diag.level = crate::Level::Warning;
                }
                FutureBreakageItem { diagnostic: Diagnostic::from_errors_diagnostic(&diag, self) }
            })
            .collect();
        let report = FutureIncompatReport { future_incompat_report: data };
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", as_pretty_json(&report))
        } else {
            writeln!(&mut self.dst, "{}", as_json(&report))
        }
        .and_then(|_| self.dst.flush());
        if let Err(e) = result {
            panic!("failed to print future breakage report: {:?}", e);
        }
    }

    fn emit_unused_externs(&mut self, lint_level: &str, unused_externs: &[&str]) {
        let data = UnusedExterns { lint_level, unused_extern_names: unused_externs };
        let result = if self.pretty {
            writeln!(&mut self.dst, "{}", as_pretty_json(&data))
        } else {
            writeln!(&mut self.dst, "{}", as_json(&data))
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

// NOTE: this has a manual implementation of Encodable which needs to be updated in
// parallel.
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
    /// Extra tool metadata
    tool_metadata: ToolMetadata,
}

macro_rules! encode_fields {
    (
        $enc:expr,                  // encoder
        $idx:expr,                  // starting field index
        $struct:expr,               // struct we're serializing
        $struct_name:ident,         // struct name
        [ $($name:ident),+$(,)? ],  // fields to encode
        [ $($ignore:ident),+$(,)? ] // fields we're skipping
    ) => {
        {
            // Pattern match to make sure all fields are accounted for
            let $struct_name { $($name,)+ $($ignore: _,)+ } = $struct;
            let mut idx = $idx;
            $(
                $enc.emit_struct_field(
                    stringify!($name),
                    idx == 0,
                    |enc| $name.encode(enc),
                )?;
                idx += 1;
            )+
            idx
        }
    };
}

// Special-case encoder to skip tool_metadata if not set
impl<E: Encoder> Encodable<E> for Diagnostic {
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_struct(false, |s| {
            let mut idx = 0;

            idx = encode_fields!(
                s,
                idx,
                self,
                Self,
                [message, code, level, spans, children, rendered],
                [tool_metadata]
            );
            if self.tool_metadata.is_set() {
                idx = encode_fields!(
                    s,
                    idx,
                    self,
                    Self,
                    [tool_metadata],
                    [message, code, level, spans, children, rendered]
                );
            }

            let _ = idx;
            Ok(())
        })
    }
}

#[derive(Encodable)]
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

#[derive(Encodable)]
struct DiagnosticSpanLine {
    text: String,

    /// 1-based, character offset in self.text.
    highlight_start: usize,

    highlight_end: usize,
}

#[derive(Encodable)]
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

#[derive(Encodable)]
struct DiagnosticCode {
    /// The code itself.
    code: String,
    /// An explanation for the code.
    explanation: Option<&'static str>,
}

#[derive(Encodable)]
struct ArtifactNotification<'a> {
    /// The path of the artifact.
    artifact: &'a Path,
    /// What kind of artifact we're emitting.
    emit: &'a str,
}

#[derive(Encodable)]
struct FutureBreakageItem {
    diagnostic: Diagnostic,
}

#[derive(Encodable)]
struct FutureIncompatReport {
    future_incompat_report: Vec<FutureBreakageItem>,
}

// NOTE: Keep this in sync with the equivalent structs in rustdoc's
// doctest component (as well as cargo).
// We could unify this struct the one in rustdoc but they have different
// ownership semantics, so doing so would create wasteful allocations.
#[derive(Encodable)]
struct UnusedExterns<'a, 'b, 'c> {
    /// The severity level of the unused dependencies lint
    lint_level: &'a str,
    /// List of unused externs by their names.
    unused_extern_names: &'b [&'c str],
}

impl Diagnostic {
    fn from_errors_diagnostic(diag: &crate::Diagnostic, je: &JsonEmitter) -> Diagnostic {
        let sugg = diag.suggestions.iter().map(|sugg| Diagnostic {
            message: sugg.msg.clone(),
            code: None,
            level: "help",
            spans: DiagnosticSpan::from_suggestion(sugg, je),
            children: vec![],
            rendered: None,
            tool_metadata: sugg.tool_metadata.clone(),
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
                false,
                je.terminal_width,
                je.macro_backtrace,
            )
            .ui_testing(je.ui_testing)
            .emit_diagnostic(diag);
        let output = Arc::try_unwrap(output.0).unwrap().into_inner().unwrap();
        let output = String::from_utf8(output).unwrap();

        Diagnostic {
            message: diag.message(),
            code: DiagnosticCode::map_opt_string(diag.code.clone(), je),
            level: diag.level.to_str(),
            spans: DiagnosticSpan::from_multispan(&diag.span, je),
            children: diag
                .children
                .iter()
                .map(|c| Diagnostic::from_sub_diagnostic(c, je))
                .chain(sugg)
                .collect(),
            rendered: Some(output),
            tool_metadata: ToolMetadata::default(),
        }
    }

    fn from_sub_diagnostic(diag: &SubDiagnostic, je: &JsonEmitter) -> Diagnostic {
        Diagnostic {
            message: diag.message(),
            code: None,
            level: diag.level.to_str(),
            spans: diag
                .render_span
                .as_ref()
                .map(|sp| DiagnosticSpan::from_multispan(sp, je))
                .unwrap_or_else(|| DiagnosticSpan::from_multispan(&diag.span, je)),
            children: vec![],
            rendered: None,
            tool_metadata: ToolMetadata::default(),
        }
    }
}

impl DiagnosticSpan {
    fn from_span_label(
        span: SpanLabel,
        suggestion: Option<(&String, Applicability)>,
        je: &JsonEmitter,
    ) -> DiagnosticSpan {
        Self::from_span_etc(span.span, span.is_primary, span.label, suggestion, je)
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
            let def_site_span =
                Self::from_span_full(bt.def_site, false, None, None, vec![].into_iter(), je);
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

    fn from_multispan(msp: &MultiSpan, je: &JsonEmitter) -> Vec<DiagnosticSpan> {
        msp.span_labels()
            .into_iter()
            .map(|span_str| Self::from_span_label(span_str, None, je))
            .collect()
    }

    fn from_suggestion(suggestion: &CodeSuggestion, je: &JsonEmitter) -> Vec<DiagnosticSpan> {
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

            DiagnosticCode { code: s, explanation: je_result.unwrap_or(None) }
        })
    }
}
