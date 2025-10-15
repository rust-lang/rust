//! The current rustc diagnostics emitter.
//!
//! An `Emitter` takes care of generating the output from a `Diag` struct.
//!
//! There are various `Emitter` implementations that generate different output formats such as
//! JSON and human readable output.
//!
//! The output types are defined in `rustc_session::config::ErrorOutputType`.

use std::borrow::Cow;
use std::cmp::{Reverse, max, min};
use std::error::Report;
use std::io::prelude::*;
use std::io::{self, IsTerminal};
use std::iter;
use std::path::Path;
use std::sync::Arc;

use derive_setters::Setters;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::sync::{DynSend, IntoDynSyncSend};
use rustc_error_messages::{FluentArgs, SpanLabel};
use rustc_lexer;
use rustc_lint_defs::pluralize;
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::source_map::SourceMap;
use rustc_span::{FileLines, FileName, SourceFile, Span, char_width, str_width};
use termcolor::{Buffer, BufferWriter, Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use tracing::{debug, instrument, trace, warn};

use crate::registry::Registry;
use crate::snippet::{
    Annotation, AnnotationColumn, AnnotationType, Line, MultilineAnnotation, Style, StyledString,
};
use crate::styled_buffer::StyledBuffer;
use crate::timings::TimingRecord;
use crate::translation::{Translator, to_fluent_args};
use crate::{
    CodeSuggestion, DiagInner, DiagMessage, ErrCode, Level, MultiSpan, Subdiag,
    SubstitutionHighlight, SuggestionStyle, TerminalUrl,
};

/// Default column width, used in tests and when terminal dimensions cannot be determined.
const DEFAULT_COLUMN_WIDTH: usize = 140;

/// Describes the way the content of the `rendered` field of the json output is generated
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanReadableErrorType {
    Default,
    Unicode,
    AnnotateSnippet,
    Short,
}

impl HumanReadableErrorType {
    pub fn short(&self) -> bool {
        *self == HumanReadableErrorType::Short
    }
}

#[derive(Clone, Copy, Debug)]
struct Margin {
    /// The available whitespace in the left that can be consumed when centering.
    pub whitespace_left: usize,
    /// The column of the beginning of leftmost span.
    pub span_left: usize,
    /// The column of the end of rightmost span.
    pub span_right: usize,
    /// The beginning of the line to be displayed.
    pub computed_left: usize,
    /// The end of the line to be displayed.
    pub computed_right: usize,
    /// The current width of the terminal. Uses value of `DEFAULT_COLUMN_WIDTH` constant by default
    /// and in tests.
    pub column_width: usize,
    /// The end column of a span label, including the span. Doesn't account for labels not in the
    /// same line as the span.
    pub label_right: usize,
}

impl Margin {
    fn new(
        whitespace_left: usize,
        span_left: usize,
        span_right: usize,
        label_right: usize,
        column_width: usize,
        max_line_len: usize,
    ) -> Self {
        // The 6 is padding to give a bit of room for `...` when displaying:
        // ```
        // error: message
        //   --> file.rs:16:58
        //    |
        // 16 | ... fn foo(self) -> Self::Bar {
        //    |                     ^^^^^^^^^
        // ```

        let mut m = Margin {
            whitespace_left: whitespace_left.saturating_sub(6),
            span_left: span_left.saturating_sub(6),
            span_right: span_right + 6,
            computed_left: 0,
            computed_right: 0,
            column_width,
            label_right: label_right + 6,
        };
        m.compute(max_line_len);
        m
    }

    fn was_cut_left(&self) -> bool {
        self.computed_left > 0
    }

    fn compute(&mut self, max_line_len: usize) {
        // When there's a lot of whitespace (>20), we want to trim it as it is useless.
        // FIXME: this doesn't account for '\t', but to do so correctly we need to perform that
        // calculation later, right before printing in order to be accurate with both unicode
        // handling and trimming of long lines.
        self.computed_left = if self.whitespace_left > 20 {
            self.whitespace_left - 16 // We want some padding.
        } else {
            0
        };
        // We want to show as much as possible, max_line_len is the rightmost boundary for the
        // relevant code.
        self.computed_right = max(max_line_len, self.computed_left);

        if self.computed_right - self.computed_left > self.column_width {
            // Trimming only whitespace isn't enough, let's get craftier.
            if self.label_right - self.whitespace_left <= self.column_width {
                // Attempt to fit the code window only trimming whitespace.
                self.computed_left = self.whitespace_left;
                self.computed_right = self.computed_left + self.column_width;
            } else if self.label_right - self.span_left <= self.column_width {
                // Attempt to fit the code window considering only the spans and labels.
                let padding_left = (self.column_width - (self.label_right - self.span_left)) / 2;
                self.computed_left = self.span_left.saturating_sub(padding_left);
                self.computed_right = self.computed_left + self.column_width;
            } else if self.span_right - self.span_left <= self.column_width {
                // Attempt to fit the code window considering the spans and labels plus padding.
                let padding_left = (self.column_width - (self.span_right - self.span_left)) / 5 * 2;
                self.computed_left = self.span_left.saturating_sub(padding_left);
                self.computed_right = self.computed_left + self.column_width;
            } else {
                // Mostly give up but still don't show the full line.
                self.computed_left = self.span_left;
                self.computed_right = self.span_right;
            }
        }
    }

    fn left(&self, line_len: usize) -> usize {
        min(self.computed_left, line_len)
    }

    fn right(&self, line_len: usize) -> usize {
        if line_len.saturating_sub(self.computed_left) <= self.column_width {
            line_len
        } else {
            min(line_len, self.computed_right)
        }
    }
}

pub enum TimingEvent {
    Start,
    End,
}

const ANONYMIZED_LINE_NUM: &str = "LL";

pub type DynEmitter = dyn Emitter + DynSend;

/// Emitter trait for emitting errors and other structured information.
pub trait Emitter {
    /// Emit a structured diagnostic.
    fn emit_diagnostic(&mut self, diag: DiagInner, registry: &Registry);

    /// Emit a notification that an artifact has been output.
    /// Currently only supported for the JSON format.
    fn emit_artifact_notification(&mut self, _path: &Path, _artifact_type: &str) {}

    /// Emit a timestamp with start/end of a timing section.
    /// Currently only supported for the JSON format.
    fn emit_timing_section(&mut self, _record: TimingRecord, _event: TimingEvent) {}

    /// Emit a report about future breakage.
    /// Currently only supported for the JSON format.
    fn emit_future_breakage_report(&mut self, _diags: Vec<DiagInner>, _registry: &Registry) {}

    /// Emit list of unused externs.
    /// Currently only supported for the JSON format.
    fn emit_unused_externs(
        &mut self,
        _lint_level: rustc_lint_defs::Level,
        _unused_externs: &[&str],
    ) {
    }

    /// Checks if should show explanations about "rustc --explain"
    fn should_show_explain(&self) -> bool {
        true
    }

    /// Checks if we can use colors in the current output stream.
    fn supports_color(&self) -> bool {
        false
    }

    fn source_map(&self) -> Option<&SourceMap>;

    fn translator(&self) -> &Translator;

    /// Formats the substitutions of the primary_span
    ///
    /// There are a lot of conditions to this method, but in short:
    ///
    /// * If the current `DiagInner` has only one visible `CodeSuggestion`,
    ///   we format the `help` suggestion depending on the content of the
    ///   substitutions. In that case, we modify the span and clear the
    ///   suggestions.
    ///
    /// * If the current `DiagInner` has multiple suggestions,
    ///   we leave `primary_span` and the suggestions untouched.
    fn primary_span_formatted(
        &self,
        primary_span: &mut MultiSpan,
        suggestions: &mut Vec<CodeSuggestion>,
        fluent_args: &FluentArgs<'_>,
    ) {
        if let Some((sugg, rest)) = suggestions.split_first() {
            let msg = self
                .translator()
                .translate_message(&sugg.msg, fluent_args)
                .map_err(Report::new)
                .unwrap();
            if rest.is_empty()
               // ^ if there is only one suggestion
               // don't display multi-suggestions as labels
               && let [substitution] = sugg.substitutions.as_slice()
               // don't display multipart suggestions as labels
               && let [part] = substitution.parts.as_slice()
               // don't display long messages as labels
               && msg.split_whitespace().count() < 10
               // don't display multiline suggestions as labels
               && !part.snippet.contains('\n')
               && ![
                    // when this style is set we want the suggestion to be a message, not inline
                    SuggestionStyle::HideCodeAlways,
                    // trivial suggestion for tooling's sake, never shown
                    SuggestionStyle::CompletelyHidden,
                    // subtle suggestion, never shown inline
                    SuggestionStyle::ShowAlways,
               ].contains(&sugg.style)
            {
                let snippet = part.snippet.trim();
                let msg = if snippet.is_empty() || sugg.style.hide_inline() {
                    // This substitution is only removal OR we explicitly don't want to show the
                    // code inline (`hide_inline`). Therefore, we don't show the substitution.
                    format!("help: {msg}")
                } else {
                    // Show the default suggestion text with the substitution
                    let confusion_type = self
                        .source_map()
                        .map(|sm| detect_confusion_type(sm, snippet, part.span))
                        .unwrap_or(ConfusionType::None);
                    format!("help: {}{}: `{}`", msg, confusion_type.label_text(), snippet,)
                };
                primary_span.push_span_label(part.span, msg);

                // We return only the modified primary_span
                suggestions.clear();
            } else {
                // if there are multiple suggestions, print them all in full
                // to be consistent. We could try to figure out if we can
                // make one (or the first one) inline, but that would give
                // undue importance to a semi-random suggestion
            }
        } else {
            // do nothing
        }
    }

    fn fix_multispans_in_extern_macros_and_render_macro_backtrace(
        &self,
        span: &mut MultiSpan,
        children: &mut Vec<Subdiag>,
        level: &Level,
        backtrace: bool,
    ) {
        // Check for spans in macros, before `fix_multispans_in_extern_macros`
        // has a chance to replace them.
        let has_macro_spans: Vec<_> = iter::once(&*span)
            .chain(children.iter().map(|child| &child.span))
            .flat_map(|span| span.primary_spans())
            .flat_map(|sp| sp.macro_backtrace())
            .filter_map(|expn_data| {
                match expn_data.kind {
                    ExpnKind::Root => None,

                    // Skip past non-macro entries, just in case there
                    // are some which do actually involve macros.
                    ExpnKind::Desugaring(..) | ExpnKind::AstPass(..) => None,

                    ExpnKind::Macro(macro_kind, name) => {
                        Some((macro_kind, name, expn_data.hide_backtrace))
                    }
                }
            })
            .collect();

        if !backtrace {
            self.fix_multispans_in_extern_macros(span, children);
        }

        self.render_multispans_macro_backtrace(span, children, backtrace);

        if !backtrace {
            // Skip builtin macros, as their expansion isn't relevant to the end user. This includes
            // actual intrinsics, like `asm!`.
            if let Some((macro_kind, name, _)) = has_macro_spans.first()
                && let Some((_, _, false)) = has_macro_spans.last()
            {
                // Mark the actual macro this originates from
                let and_then = if let Some((macro_kind, last_name, _)) = has_macro_spans.last()
                    && last_name != name
                {
                    let descr = macro_kind.descr();
                    format!(" which comes from the expansion of the {descr} `{last_name}`")
                } else {
                    "".to_string()
                };

                let descr = macro_kind.descr();
                let msg = format!(
                    "this {level} originates in the {descr} `{name}`{and_then} \
                    (in Nightly builds, run with -Z macro-backtrace for more info)",
                );

                children.push(Subdiag {
                    level: Level::Note,
                    messages: vec![(DiagMessage::from(msg), Style::NoStyle)],
                    span: MultiSpan::new(),
                });
            }
        }
    }

    fn render_multispans_macro_backtrace(
        &self,
        span: &mut MultiSpan,
        children: &mut Vec<Subdiag>,
        backtrace: bool,
    ) {
        for span in iter::once(span).chain(children.iter_mut().map(|child| &mut child.span)) {
            self.render_multispan_macro_backtrace(span, backtrace);
        }
    }

    fn render_multispan_macro_backtrace(&self, span: &mut MultiSpan, always_backtrace: bool) {
        let mut new_labels = FxIndexSet::default();

        for &sp in span.primary_spans() {
            if sp.is_dummy() {
                continue;
            }

            // FIXME(eddyb) use `retain` on `macro_backtrace` to remove all the
            // entries we don't want to print, to make sure the indices being
            // printed are contiguous (or omitted if there's only one entry).
            let macro_backtrace: Vec<_> = sp.macro_backtrace().collect();
            for (i, trace) in macro_backtrace.iter().rev().enumerate() {
                if trace.def_site.is_dummy() {
                    continue;
                }

                if always_backtrace {
                    new_labels.insert((
                        trace.def_site,
                        format!(
                            "in this expansion of `{}`{}",
                            trace.kind.descr(),
                            if macro_backtrace.len() > 1 {
                                // if macro_backtrace.len() == 1 it'll be
                                // pointed at by "in this macro invocation"
                                format!(" (#{})", i + 1)
                            } else {
                                String::new()
                            },
                        ),
                    ));
                }

                // Don't add a label on the call site if the diagnostic itself
                // already points to (a part of) that call site, as the label
                // is meant for showing the relevant invocation when the actual
                // diagnostic is pointing to some part of macro definition.
                //
                // This also handles the case where an external span got replaced
                // with the call site span by `fix_multispans_in_extern_macros`.
                //
                // NB: `-Zmacro-backtrace` overrides this, for uniformity, as the
                // "in this expansion of" label above is always added in that mode,
                // and it needs an "in this macro invocation" label to match that.
                let redundant_span = trace.call_site.contains(sp);

                if !redundant_span || always_backtrace {
                    let msg: Cow<'static, _> = match trace.kind {
                        ExpnKind::Macro(MacroKind::Attr, _) => {
                            "this attribute macro expansion".into()
                        }
                        ExpnKind::Macro(MacroKind::Derive, _) => {
                            "this derive macro expansion".into()
                        }
                        ExpnKind::Macro(MacroKind::Bang, _) => "this macro invocation".into(),
                        ExpnKind::Root => "the crate root".into(),
                        ExpnKind::AstPass(kind) => kind.descr().into(),
                        ExpnKind::Desugaring(kind) => {
                            format!("this {} desugaring", kind.descr()).into()
                        }
                    };
                    new_labels.insert((
                        trace.call_site,
                        format!(
                            "in {}{}",
                            msg,
                            if macro_backtrace.len() > 1 && always_backtrace {
                                // only specify order when the macro
                                // backtrace is multiple levels deep
                                format!(" (#{})", i + 1)
                            } else {
                                String::new()
                            },
                        ),
                    ));
                }
                if !always_backtrace {
                    break;
                }
            }
        }

        for (label_span, label_text) in new_labels {
            span.push_span_label(label_span, label_text);
        }
    }

    // This does a small "fix" for multispans by looking to see if it can find any that
    // point directly at external macros. Since these are often difficult to read,
    // this will change the span to point at the use site.
    fn fix_multispans_in_extern_macros(&self, span: &mut MultiSpan, children: &mut Vec<Subdiag>) {
        debug!("fix_multispans_in_extern_macros: before: span={:?} children={:?}", span, children);
        self.fix_multispan_in_extern_macros(span);
        for child in children.iter_mut() {
            self.fix_multispan_in_extern_macros(&mut child.span);
        }
        debug!("fix_multispans_in_extern_macros: after: span={:?} children={:?}", span, children);
    }

    // This "fixes" MultiSpans that contain `Span`s pointing to locations inside of external macros.
    // Since these locations are often difficult to read,
    // we move these spans from the external macros to their corresponding use site.
    fn fix_multispan_in_extern_macros(&self, span: &mut MultiSpan) {
        let Some(source_map) = self.source_map() else { return };
        // First, find all the spans in external macros and point instead at their use site.
        let replacements: Vec<(Span, Span)> = span
            .primary_spans()
            .iter()
            .copied()
            .chain(span.span_labels().iter().map(|sp_label| sp_label.span))
            .filter_map(|sp| {
                if !sp.is_dummy() && source_map.is_imported(sp) {
                    let maybe_callsite = sp.source_callsite();
                    if sp != maybe_callsite {
                        return Some((sp, maybe_callsite));
                    }
                }
                None
            })
            .collect();

        // After we have them, make sure we replace these 'bad' def sites with their use sites.
        for (from, to) in replacements {
            span.replace(from, to);
        }
    }
}

impl Emitter for HumanEmitter {
    fn source_map(&self) -> Option<&SourceMap> {
        self.sm.as_deref()
    }

    fn emit_diagnostic(&mut self, mut diag: DiagInner, _registry: &Registry) {
        let fluent_args = to_fluent_args(diag.args.iter());

        if self.track_diagnostics && diag.span.has_primary_spans() && !diag.span.is_dummy() {
            diag.children.insert(0, diag.emitted_at_sub_diag());
        }

        let mut suggestions = diag.suggestions.unwrap_tag();
        self.primary_span_formatted(&mut diag.span, &mut suggestions, &fluent_args);

        self.fix_multispans_in_extern_macros_and_render_macro_backtrace(
            &mut diag.span,
            &mut diag.children,
            &diag.level,
            self.macro_backtrace,
        );

        self.emit_messages_default(
            &diag.level,
            &diag.messages,
            &fluent_args,
            &diag.code,
            &diag.span,
            &diag.children,
            &suggestions,
        );
    }

    fn should_show_explain(&self) -> bool {
        !self.short_message
    }

    fn supports_color(&self) -> bool {
        self.dst.supports_color()
    }

    fn translator(&self) -> &Translator {
        &self.translator
    }
}

/// An emitter that does nothing when emitting a non-fatal diagnostic.
/// Fatal diagnostics are forwarded to `fatal_emitter` to avoid silent
/// failures of rustc, as witnessed e.g. in issue #89358.
pub struct FatalOnlyEmitter {
    pub fatal_emitter: Box<dyn Emitter + DynSend>,
    pub fatal_note: Option<String>,
}

impl Emitter for FatalOnlyEmitter {
    fn source_map(&self) -> Option<&SourceMap> {
        None
    }

    fn emit_diagnostic(&mut self, mut diag: DiagInner, registry: &Registry) {
        if diag.level == Level::Fatal {
            if let Some(fatal_note) = &self.fatal_note {
                diag.sub(Level::Note, fatal_note.clone(), MultiSpan::new());
            }
            self.fatal_emitter.emit_diagnostic(diag, registry);
        }
    }

    fn translator(&self) -> &Translator {
        self.fatal_emitter.translator()
    }
}

pub struct SilentEmitter {
    pub translator: Translator,
}

impl Emitter for SilentEmitter {
    fn source_map(&self) -> Option<&SourceMap> {
        None
    }

    fn emit_diagnostic(&mut self, _diag: DiagInner, _registry: &Registry) {}

    fn translator(&self) -> &Translator {
        &self.translator
    }
}

/// Maximum number of suggestions to be shown
///
/// Arbitrary, but taken from trait import suggestion limit
pub const MAX_SUGGESTIONS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorConfig {
    Auto,
    Always,
    Never,
}

impl ColorConfig {
    pub fn to_color_choice(self) -> ColorChoice {
        match self {
            ColorConfig::Always => {
                if io::stderr().is_terminal() {
                    ColorChoice::Always
                } else {
                    ColorChoice::AlwaysAnsi
                }
            }
            ColorConfig::Never => ColorChoice::Never,
            ColorConfig::Auto if io::stderr().is_terminal() => ColorChoice::Auto,
            ColorConfig::Auto => ColorChoice::Never,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTheme {
    Ascii,
    Unicode,
}

/// Handles the writing of `HumanReadableErrorType::Default` and `HumanReadableErrorType::Short`
#[derive(Setters)]
pub struct HumanEmitter {
    #[setters(skip)]
    dst: IntoDynSyncSend<Destination>,
    sm: Option<Arc<SourceMap>>,
    #[setters(skip)]
    translator: Translator,
    short_message: bool,
    ui_testing: bool,
    ignored_directories_in_source_blocks: Vec<String>,
    diagnostic_width: Option<usize>,

    macro_backtrace: bool,
    track_diagnostics: bool,
    terminal_url: TerminalUrl,
    theme: OutputTheme,
}

#[derive(Debug)]
pub(crate) struct FileWithAnnotatedLines {
    pub(crate) file: Arc<SourceFile>,
    pub(crate) lines: Vec<Line>,
    multiline_depth: usize,
}

impl HumanEmitter {
    pub fn new(dst: Destination, translator: Translator) -> HumanEmitter {
        HumanEmitter {
            dst: IntoDynSyncSend(dst),
            sm: None,
            translator,
            short_message: false,
            ui_testing: false,
            ignored_directories_in_source_blocks: Vec::new(),
            diagnostic_width: None,
            macro_backtrace: false,
            track_diagnostics: false,
            terminal_url: TerminalUrl::No,
            theme: OutputTheme::Ascii,
        }
    }

    fn maybe_anonymized(&self, line_num: usize) -> Cow<'static, str> {
        if self.ui_testing {
            Cow::Borrowed(ANONYMIZED_LINE_NUM)
        } else {
            Cow::Owned(line_num.to_string())
        }
    }

    fn draw_line(
        &self,
        buffer: &mut StyledBuffer,
        source_string: &str,
        line_index: usize,
        line_offset: usize,
        width_offset: usize,
        code_offset: usize,
        margin: Margin,
    ) -> usize {
        let line_len = source_string.len();
        // Create the source line we will highlight.
        let left = margin.left(line_len);
        let right = margin.right(line_len);
        // FIXME: The following code looks fishy. See #132860.
        // On long lines, we strip the source line, accounting for unicode.
        let code: String = source_string
            .chars()
            .enumerate()
            .skip_while(|(i, _)| *i < left)
            .take_while(|(i, _)| *i < right)
            .map(|(_, c)| c)
            .collect();
        let code = normalize_whitespace(&code);
        let was_cut_right =
            source_string.chars().enumerate().skip_while(|(i, _)| *i < right).next().is_some();
        buffer.puts(line_offset, code_offset, &code, Style::Quotation);
        let placeholder = self.margin();
        if margin.was_cut_left() {
            // We have stripped some code/whitespace from the beginning, make it clear.
            buffer.puts(line_offset, code_offset, placeholder, Style::LineNumber);
        }
        if was_cut_right {
            let padding = str_width(placeholder);
            // We have stripped some code after the rightmost span end, make it clear we did so.
            buffer.puts(
                line_offset,
                code_offset + str_width(&code) - padding,
                placeholder,
                Style::LineNumber,
            );
        }
        self.draw_line_num(buffer, line_index, line_offset, width_offset - 3);
        self.draw_col_separator_no_space(buffer, line_offset, width_offset - 2);
        left
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn render_source_line(
        &self,
        buffer: &mut StyledBuffer,
        file: Arc<SourceFile>,
        line: &Line,
        width_offset: usize,
        code_offset: usize,
        margin: Margin,
        close_window: bool,
    ) -> Vec<(usize, Style)> {
        // Draw:
        //
        //   LL | ... code ...
        //      |     ^^-^ span label
        //      |       |
        //      |       secondary span label
        //
        //   ^^ ^ ^^^ ^^^^ ^^^ we don't care about code too far to the right of a span, we trim it
        //   |  | |   |
        //   |  | |   actual code found in your source code and the spans we use to mark it
        //   |  | when there's too much wasted space to the left, trim it
        //   |  vertical divider between the column number and the code
        //   column number

        if line.line_index == 0 {
            return Vec::new();
        }

        let Some(source_string) = file.get_line(line.line_index - 1) else {
            return Vec::new();
        };
        trace!(?source_string);

        let line_offset = buffer.num_lines();

        // Left trim.
        // FIXME: This looks fishy. See #132860.
        let left = self.draw_line(
            buffer,
            &source_string,
            line.line_index,
            line_offset,
            width_offset,
            code_offset,
            margin,
        );

        // Special case when there's only one annotation involved, it is the start of a multiline
        // span and there's no text at the beginning of the code line. Instead of doing the whole
        // graph:
        //
        // 2 |   fn foo() {
        //   |  _^
        // 3 | |
        // 4 | | }
        //   | |_^ test
        //
        // we simplify the output to:
        //
        // 2 | / fn foo() {
        // 3 | |
        // 4 | | }
        //   | |_^ test
        let mut buffer_ops = vec![];
        let mut annotations = vec![];
        let mut short_start = true;
        for ann in &line.annotations {
            if let AnnotationType::MultilineStart(depth) = ann.annotation_type {
                if source_string.chars().take(ann.start_col.file).all(|c| c.is_whitespace()) {
                    let uline = self.underline(ann.is_primary);
                    let chr = uline.multiline_whole_line;
                    annotations.push((depth, uline.style));
                    buffer_ops.push((line_offset, width_offset + depth - 1, chr, uline.style));
                } else {
                    short_start = false;
                    break;
                }
            } else if let AnnotationType::MultilineLine(_) = ann.annotation_type {
            } else {
                short_start = false;
                break;
            }
        }
        if short_start {
            for (y, x, c, s) in buffer_ops {
                buffer.putc(y, x, c, s);
            }
            return annotations;
        }

        // We want to display like this:
        //
        //      vec.push(vec.pop().unwrap());
        //      ---      ^^^               - previous borrow ends here
        //      |        |
        //      |        error occurs here
        //      previous borrow of `vec` occurs here
        //
        // But there are some weird edge cases to be aware of:
        //
        //      vec.push(vec.pop().unwrap());
        //      --------                    - previous borrow ends here
        //      ||
        //      |this makes no sense
        //      previous borrow of `vec` occurs here
        //
        // For this reason, we group the lines into "highlight lines"
        // and "annotations lines", where the highlight lines have the `^`.

        // Sort the annotations by (start, end col)
        // The labels are reversed, sort and then reversed again.
        // Consider a list of annotations (A1, A2, C1, C2, B1, B2) where
        // the letter signifies the span. Here we are only sorting by the
        // span and hence, the order of the elements with the same span will
        // not change. On reversing the ordering (|a, b| but b.cmp(a)), you get
        // (C1, C2, B1, B2, A1, A2). All the elements with the same span are
        // still ordered first to last, but all the elements with different
        // spans are ordered by their spans in last to first order. Last to
        // first order is important, because the jiggly lines and | are on
        // the left, so the rightmost span needs to be rendered first,
        // otherwise the lines would end up needing to go over a message.

        let mut annotations = line.annotations.clone();
        annotations.sort_by_key(|a| Reverse(a.start_col));

        // First, figure out where each label will be positioned.
        //
        // In the case where you have the following annotations:
        //
        //      vec.push(vec.pop().unwrap());
        //      --------                    - previous borrow ends here [C]
        //      ||
        //      |this makes no sense [B]
        //      previous borrow of `vec` occurs here [A]
        //
        // `annotations_position` will hold [(2, A), (1, B), (0, C)].
        //
        // We try, when possible, to stick the rightmost annotation at the end
        // of the highlight line:
        //
        //      vec.push(vec.pop().unwrap());
        //      ---      ---               - previous borrow ends here
        //
        // But sometimes that's not possible because one of the other
        // annotations overlaps it. For example, from the test
        // `span_overlap_label`, we have the following annotations
        // (written on distinct lines for clarity):
        //
        //      fn foo(x: u32) {
        //      --------------
        //             -
        //
        // In this case, we can't stick the rightmost-most label on
        // the highlight line, or we would get:
        //
        //      fn foo(x: u32) {
        //      -------- x_span
        //      |
        //      fn_span
        //
        // which is totally weird. Instead we want:
        //
        //      fn foo(x: u32) {
        //      --------------
        //      |      |
        //      |      x_span
        //      fn_span
        //
        // which is...less weird, at least. In fact, in general, if
        // the rightmost span overlaps with any other span, we should
        // use the "hang below" version, so we can at least make it
        // clear where the span *starts*. There's an exception for this
        // logic, when the labels do not have a message:
        //
        //      fn foo(x: u32) {
        //      --------------
        //             |
        //             x_span
        //
        // instead of:
        //
        //      fn foo(x: u32) {
        //      --------------
        //      |      |
        //      |      x_span
        //      <EMPTY LINE>
        //
        let mut overlap = vec![false; annotations.len()];
        let mut annotations_position = vec![];
        let mut line_len: usize = 0;
        let mut p = 0;
        for (i, annotation) in annotations.iter().enumerate() {
            for (j, next) in annotations.iter().enumerate() {
                if overlaps(next, annotation, 0) && j > i {
                    overlap[i] = true;
                    overlap[j] = true;
                }
                if overlaps(next, annotation, 0)  // This label overlaps with another one and both
                    && annotation.has_label()     // take space (they have text and are not
                    && j > i                      // multiline lines).
                    && p == 0
                // We're currently on the first line, move the label one line down
                {
                    // If we're overlapping with an un-labelled annotation with the same span
                    // we can just merge them in the output
                    if next.start_col == annotation.start_col
                        && next.end_col == annotation.end_col
                        && !next.has_label()
                    {
                        continue;
                    }

                    // This annotation needs a new line in the output.
                    p += 1;
                    break;
                }
            }
            annotations_position.push((p, annotation));
            for (j, next) in annotations.iter().enumerate() {
                if j > i {
                    let l = next.label.as_ref().map_or(0, |label| label.len() + 2);
                    if (overlaps(next, annotation, l) // Do not allow two labels to be in the same
                                                     // line if they overlap including padding, to
                                                     // avoid situations like:
                                                     //
                                                     //      fn foo(x: u32) {
                                                     //      -------^------
                                                     //      |      |
                                                     //      fn_spanx_span
                                                     //
                        && annotation.has_label()    // Both labels must have some text, otherwise
                        && next.has_label())         // they are not overlapping.
                                                     // Do not add a new line if this annotation
                                                     // or the next are vertical line placeholders.
                        || (annotation.takes_space() // If either this or the next annotation is
                            && next.has_label())     // multiline start/end, move it to a new line
                        || (annotation.has_label()   // so as not to overlap the horizontal lines.
                            && next.takes_space())
                        || (annotation.takes_space() && next.takes_space())
                        || (overlaps(next, annotation, l)
                            && next.end_col <= annotation.end_col
                            && next.has_label()
                            && p == 0)
                    // Avoid #42595.
                    {
                        // This annotation needs a new line in the output.
                        p += 1;
                        break;
                    }
                }
            }
            line_len = max(line_len, p);
        }

        if line_len != 0 {
            line_len += 1;
        }

        // If there are no annotations or the only annotations on this line are
        // MultilineLine, then there's only code being shown, stop processing.
        if line.annotations.iter().all(|a| a.is_line()) {
            return vec![];
        }

        if annotations_position
            .iter()
            .all(|(_, ann)| matches!(ann.annotation_type, AnnotationType::MultilineStart(_)))
            && let Some(max_pos) = annotations_position.iter().map(|(pos, _)| *pos).max()
        {
            // Special case the following, so that we minimize overlapping multiline spans.
            //
            // 3 │       X0 Y0 Z0
            //   │ ┏━━━━━┛  │  │     < We are writing these lines
            //   │ ┃┌───────┘  │     < by reverting the "depth" of
            //   │ ┃│┌─────────┘     < their multiline spans.
            // 4 │ ┃││   X1 Y1 Z1
            // 5 │ ┃││   X2 Y2 Z2
            //   │ ┃│└────╿──│──┘ `Z` label
            //   │ ┃└─────│──┤
            //   │ ┗━━━━━━┥  `Y` is a good letter too
            //   ╰╴       `X` is a good letter
            for (pos, _) in &mut annotations_position {
                *pos = max_pos - *pos;
            }
            // We know then that we don't need an additional line for the span label, saving us
            // one line of vertical space.
            line_len = line_len.saturating_sub(1);
        }

        // Write the column separator.
        //
        // After this we will have:
        //
        // 2 |   fn foo() {
        //   |
        //   |
        //   |
        // 3 |
        // 4 |   }
        //   |
        for pos in 0..=line_len {
            self.draw_col_separator_no_space(buffer, line_offset + pos + 1, width_offset - 2);
        }
        if close_window {
            self.draw_col_separator_end(buffer, line_offset + line_len + 1, width_offset - 2);
        }

        // Write the horizontal lines for multiline annotations
        // (only the first and last lines need this).
        //
        // After this we will have:
        //
        // 2 |   fn foo() {
        //   |  __________
        //   |
        //   |
        // 3 |
        // 4 |   }
        //   |  _
        for &(pos, annotation) in &annotations_position {
            let underline = self.underline(annotation.is_primary);
            let pos = pos + 1;
            match annotation.annotation_type {
                AnnotationType::MultilineStart(depth) | AnnotationType::MultilineEnd(depth) => {
                    let pre: usize = source_string
                        .chars()
                        .take(annotation.start_col.file)
                        .skip(left)
                        .map(|c| char_width(c))
                        .sum();
                    self.draw_range(
                        buffer,
                        underline.multiline_horizontal,
                        line_offset + pos,
                        width_offset + depth,
                        code_offset + pre,
                        underline.style,
                    );
                }
                _ => {}
            }
        }

        // Write the vertical lines for labels that are on a different line as the underline.
        //
        // After this we will have:
        //
        // 2 |   fn foo() {
        //   |  __________
        //   | |    |
        //   | |
        // 3 | |
        // 4 | | }
        //   | |_
        for &(pos, annotation) in &annotations_position {
            let underline = self.underline(annotation.is_primary);
            let pos = pos + 1;

            let code_offset = code_offset
                + source_string
                    .chars()
                    .take(annotation.start_col.file)
                    .skip(left)
                    .map(|c| char_width(c))
                    .sum::<usize>();
            if pos > 1 && (annotation.has_label() || annotation.takes_space()) {
                for p in line_offset + 1..=line_offset + pos {
                    buffer.putc(
                        p,
                        code_offset,
                        match annotation.annotation_type {
                            AnnotationType::MultilineLine(_) => underline.multiline_vertical,
                            _ => underline.vertical_text_line,
                        },
                        underline.style,
                    );
                }
                if let AnnotationType::MultilineStart(_) = annotation.annotation_type {
                    buffer.putc(
                        line_offset + pos,
                        code_offset,
                        underline.bottom_right,
                        underline.style,
                    );
                }
                if let AnnotationType::MultilineEnd(_) = annotation.annotation_type
                    && annotation.has_label()
                {
                    buffer.putc(
                        line_offset + pos,
                        code_offset,
                        underline.multiline_bottom_right_with_text,
                        underline.style,
                    );
                }
            }
            match annotation.annotation_type {
                AnnotationType::MultilineStart(depth) => {
                    buffer.putc(
                        line_offset + pos,
                        width_offset + depth - 1,
                        underline.top_left,
                        underline.style,
                    );
                    for p in line_offset + pos + 1..line_offset + line_len + 2 {
                        buffer.putc(
                            p,
                            width_offset + depth - 1,
                            underline.multiline_vertical,
                            underline.style,
                        );
                    }
                }
                AnnotationType::MultilineEnd(depth) => {
                    for p in line_offset..line_offset + pos {
                        buffer.putc(
                            p,
                            width_offset + depth - 1,
                            underline.multiline_vertical,
                            underline.style,
                        );
                    }
                    buffer.putc(
                        line_offset + pos,
                        width_offset + depth - 1,
                        underline.bottom_left,
                        underline.style,
                    );
                }
                _ => (),
            }
        }

        // Write the labels on the annotations that actually have a label.
        //
        // After this we will have:
        //
        // 2 |   fn foo() {
        //   |  __________
        //   |      |
        //   |      something about `foo`
        // 3 |
        // 4 |   }
        //   |  _  test
        for &(pos, annotation) in &annotations_position {
            let style =
                if annotation.is_primary { Style::LabelPrimary } else { Style::LabelSecondary };
            let (pos, col) = if pos == 0 {
                let pre: usize = source_string
                    .chars()
                    .take(annotation.end_col.file)
                    .skip(left)
                    .map(|c| char_width(c))
                    .sum();
                if annotation.end_col.file == 0 {
                    (pos + 1, (pre + 2))
                } else {
                    let pad = if annotation.end_col.file - annotation.start_col.file == 0 {
                        2
                    } else {
                        1
                    };
                    (pos + 1, (pre + pad))
                }
            } else {
                let pre: usize = source_string
                    .chars()
                    .take(annotation.start_col.file)
                    .skip(left)
                    .map(|c| char_width(c))
                    .sum();
                (pos + 2, pre)
            };
            if let Some(ref label) = annotation.label {
                buffer.puts(line_offset + pos, code_offset + col, label, style);
            }
        }

        // Sort from biggest span to smallest span so that smaller spans are
        // represented in the output:
        //
        // x | fn foo()
        //   | ^^^---^^
        //   | |  |
        //   | |  something about `foo`
        //   | something about `fn foo()`
        annotations_position.sort_by_key(|(_, ann)| {
            // Decreasing order. When annotations share the same length, prefer `Primary`.
            (Reverse(ann.len()), ann.is_primary)
        });

        // Write the underlines.
        //
        // After this we will have:
        //
        // 2 |   fn foo() {
        //   |  ____-_____^
        //   |      |
        //   |      something about `foo`
        // 3 |
        // 4 |   }
        //   |  _^  test
        for &(pos, annotation) in &annotations_position {
            let uline = self.underline(annotation.is_primary);
            let width = annotation.end_col.file - annotation.start_col.file;
            let previous: String =
                source_string.chars().take(annotation.start_col.file).skip(left).collect();
            let underlined: String =
                source_string.chars().skip(annotation.start_col.file).take(width).collect();
            debug!(?previous, ?underlined);
            let code_offset = code_offset
                + source_string
                    .chars()
                    .take(annotation.start_col.file)
                    .skip(left)
                    .map(|c| char_width(c))
                    .sum::<usize>();
            let ann_width: usize = source_string
                .chars()
                .skip(annotation.start_col.file)
                .take(width)
                .map(|c| char_width(c))
                .sum();
            let ann_width = if ann_width == 0
                && matches!(annotation.annotation_type, AnnotationType::Singleline)
            {
                1
            } else {
                ann_width
            };
            for p in 0..ann_width {
                // The default span label underline.
                buffer.putc(line_offset + 1, code_offset + p, uline.underline, uline.style);
            }

            if pos == 0
                && matches!(
                    annotation.annotation_type,
                    AnnotationType::MultilineStart(_) | AnnotationType::MultilineEnd(_)
                )
            {
                // The beginning of a multiline span with its leftward moving line on the same line.
                buffer.putc(
                    line_offset + 1,
                    code_offset,
                    match annotation.annotation_type {
                        AnnotationType::MultilineStart(_) => uline.top_right_flat,
                        AnnotationType::MultilineEnd(_) => uline.multiline_end_same_line,
                        _ => panic!("unexpected annotation type: {annotation:?}"),
                    },
                    uline.style,
                );
            } else if pos != 0
                && matches!(
                    annotation.annotation_type,
                    AnnotationType::MultilineStart(_) | AnnotationType::MultilineEnd(_)
                )
            {
                // The beginning of a multiline span with its leftward moving line on another line,
                // so we start going down first.
                buffer.putc(
                    line_offset + 1,
                    code_offset,
                    match annotation.annotation_type {
                        AnnotationType::MultilineStart(_) => uline.multiline_start_down,
                        AnnotationType::MultilineEnd(_) => uline.multiline_end_up,
                        _ => panic!("unexpected annotation type: {annotation:?}"),
                    },
                    uline.style,
                );
            } else if pos != 0 && annotation.has_label() {
                // The beginning of a span label with an actual label, we'll point down.
                buffer.putc(line_offset + 1, code_offset, uline.label_start, uline.style);
            }
        }

        // We look for individual *long* spans, and we trim the *middle*, so that we render
        // LL | ...= [0, 0, 0, ..., 0, 0];
        //    |      ^^^^^^^^^^...^^^^^^^ expected `&[u8]`, found `[{integer}; 1680]`
        for (i, (_pos, annotation)) in annotations_position.iter().enumerate() {
            // Skip cases where multiple spans overlap each other.
            if overlap[i] {
                continue;
            };
            let AnnotationType::Singleline = annotation.annotation_type else { continue };
            let width = annotation.end_col.display - annotation.start_col.display;
            if width > margin.column_width * 2 && width > 10 {
                // If the terminal is *too* small, we keep at least a tiny bit of the span for
                // display.
                let pad = max(margin.column_width / 3, 5);
                // Code line
                buffer.replace(
                    line_offset,
                    annotation.start_col.file + pad,
                    annotation.end_col.file - pad,
                    self.margin(),
                );
                // Underline line
                buffer.replace(
                    line_offset + 1,
                    annotation.start_col.file + pad,
                    annotation.end_col.file - pad,
                    self.margin(),
                );
            }
        }
        annotations_position
            .iter()
            .filter_map(|&(_, annotation)| match annotation.annotation_type {
                AnnotationType::MultilineStart(p) | AnnotationType::MultilineEnd(p) => {
                    let style = if annotation.is_primary {
                        Style::LabelPrimary
                    } else {
                        Style::LabelSecondary
                    };
                    Some((p, style))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    }

    fn get_multispan_max_line_num(&mut self, msp: &MultiSpan) -> usize {
        let Some(ref sm) = self.sm else {
            return 0;
        };

        let will_be_emitted = |span: Span| {
            !span.is_dummy() && {
                let file = sm.lookup_source_file(span.hi());
                should_show_source_code(&self.ignored_directories_in_source_blocks, sm, &file)
            }
        };

        let mut max = 0;
        for primary_span in msp.primary_spans() {
            if will_be_emitted(*primary_span) {
                let hi = sm.lookup_char_pos(primary_span.hi());
                max = (hi.line).max(max);
            }
        }
        if !self.short_message {
            for span_label in msp.span_labels() {
                if will_be_emitted(span_label.span) {
                    let hi = sm.lookup_char_pos(span_label.span.hi());
                    max = (hi.line).max(max);
                }
            }
        }

        max
    }

    fn get_max_line_num(&mut self, span: &MultiSpan, children: &[Subdiag]) -> usize {
        let primary = self.get_multispan_max_line_num(span);
        children
            .iter()
            .map(|sub| self.get_multispan_max_line_num(&sub.span))
            .max()
            .unwrap_or(0)
            .max(primary)
    }

    /// Adds a left margin to every line but the first, given a padding length and the label being
    /// displayed, keeping the provided highlighting.
    fn msgs_to_buffer(
        &self,
        buffer: &mut StyledBuffer,
        msgs: &[(DiagMessage, Style)],
        args: &FluentArgs<'_>,
        padding: usize,
        label: &str,
        override_style: Option<Style>,
    ) -> usize {
        // The extra 5 ` ` is padding that's always needed to align to the `note: `:
        //
        //   error: message
        //     --> file.rs:13:20
        //      |
        //   13 |     <CODE>
        //      |      ^^^^
        //      |
        //      = note: multiline
        //              message
        //   ++^^^----xx
        //    |  |   | |
        //    |  |   | magic `2`
        //    |  |   length of label
        //    |  magic `3`
        //    `max_line_num_len`
        let padding = " ".repeat(padding + label.len() + 5);

        /// Returns `override` if it is present and `style` is `NoStyle` or `style` otherwise
        fn style_or_override(style: Style, override_: Option<Style>) -> Style {
            match (style, override_) {
                (Style::NoStyle, Some(override_)) => override_,
                _ => style,
            }
        }

        let mut line_number = 0;

        // Provided the following diagnostic message:
        //
        //     let msgs = vec![
        //       ("
        //       ("highlighted multiline\nstring to\nsee how it ", Style::NoStyle),
        //       ("looks", Style::Highlight),
        //       ("with\nvery ", Style::NoStyle),
        //       ("weird", Style::Highlight),
        //       (" formats\n", Style::NoStyle),
        //       ("see?", Style::Highlight),
        //     ];
        //
        // the expected output on a note is (* surround the highlighted text)
        //
        //        = note: highlighted multiline
        //                string to
        //                see how it *looks* with
        //                very *weird* formats
        //                see?
        for (text, style) in msgs.iter() {
            let text = self.translator.translate_message(text, args).map_err(Report::new).unwrap();
            let text = &normalize_whitespace(&text);
            let lines = text.split('\n').collect::<Vec<_>>();
            if lines.len() > 1 {
                for (i, line) in lines.iter().enumerate() {
                    if i != 0 {
                        line_number += 1;
                        buffer.append(line_number, &padding, Style::NoStyle);
                    }
                    buffer.append(line_number, line, style_or_override(*style, override_style));
                }
            } else {
                buffer.append(line_number, text, style_or_override(*style, override_style));
            }
        }
        line_number
    }

    #[instrument(level = "trace", skip(self, args), ret)]
    fn emit_messages_default_inner(
        &mut self,
        msp: &MultiSpan,
        msgs: &[(DiagMessage, Style)],
        args: &FluentArgs<'_>,
        code: &Option<ErrCode>,
        level: &Level,
        max_line_num_len: usize,
        is_secondary: bool,
        is_cont: bool,
    ) -> io::Result<CodeWindowStatus> {
        let mut buffer = StyledBuffer::new();

        if !msp.has_primary_spans() && !msp.has_span_labels() && is_secondary && !self.short_message
        {
            // This is a secondary message with no span info
            for _ in 0..max_line_num_len {
                buffer.prepend(0, " ", Style::NoStyle);
            }
            self.draw_note_separator(&mut buffer, 0, max_line_num_len + 1, is_cont);
            if *level != Level::FailureNote {
                buffer.append(0, level.to_str(), Style::MainHeaderMsg);
                buffer.append(0, ": ", Style::NoStyle);
            }
            let printed_lines =
                self.msgs_to_buffer(&mut buffer, msgs, args, max_line_num_len, "note", None);
            if is_cont && matches!(self.theme, OutputTheme::Unicode) {
                // There's another note after this one, associated to the subwindow above.
                // We write additional vertical lines to join them:
                //   ╭▸ test.rs:3:3
                //   │
                // 3 │   code
                //   │   ━━━━
                //   │
                //   ├ note: foo
                //   │       bar
                //   ╰ note: foo
                //           bar
                for i in 1..=printed_lines {
                    self.draw_col_separator_no_space(&mut buffer, i, max_line_num_len + 1);
                }
            }
        } else {
            let mut label_width = 0;
            // The failure note level itself does not provide any useful diagnostic information
            if *level != Level::FailureNote {
                buffer.append(0, level.to_str(), Style::Level(*level));
                label_width += level.to_str().len();
            }
            if let Some(code) = code {
                buffer.append(0, "[", Style::Level(*level));
                let code = if let TerminalUrl::Yes = self.terminal_url {
                    let path = "https://doc.rust-lang.org/error_codes";
                    format!("\x1b]8;;{path}/{code}.html\x07{code}\x1b]8;;\x07")
                } else {
                    code.to_string()
                };
                buffer.append(0, &code, Style::Level(*level));
                buffer.append(0, "]", Style::Level(*level));
                label_width += 2 + code.len();
            }
            let header_style = if is_secondary {
                Style::HeaderMsg
            } else if self.short_message {
                // For short messages avoid bolding the message, as it doesn't look great (#63835).
                Style::NoStyle
            } else {
                Style::MainHeaderMsg
            };
            if *level != Level::FailureNote {
                buffer.append(0, ": ", header_style);
                label_width += 2;
            }
            let mut line = 0;
            for (text, style) in msgs.iter() {
                let text =
                    self.translator.translate_message(text, args).map_err(Report::new).unwrap();
                // Account for newlines to align output to its label.
                for text in normalize_whitespace(&text).lines() {
                    buffer.append(
                        line,
                        &format!(
                            "{}{}",
                            if line == 0 { String::new() } else { " ".repeat(label_width) },
                            text
                        ),
                        match style {
                            Style::Highlight => *style,
                            _ => header_style,
                        },
                    );
                    line += 1;
                }
                // We add lines above, but if the last line has no explicit newline (which would
                // yield an empty line), then we revert one line up to continue with the next
                // styled text chunk on the same line as the last one from the prior one. Otherwise
                // every `text` would appear on their own line (because even though they didn't end
                // in '\n', they advanced `line` by one).
                if line > 0 {
                    line -= 1;
                }
            }
            if self.short_message {
                let labels = msp
                    .span_labels()
                    .into_iter()
                    .filter_map(|label| match label.label {
                        Some(msg) if label.is_primary => {
                            let text = self.translator.translate_message(&msg, args).ok()?;
                            if !text.trim().is_empty() { Some(text.to_string()) } else { None }
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                if !labels.is_empty() {
                    buffer.append(line, ": ", Style::NoStyle);
                    buffer.append(line, &labels, Style::NoStyle);
                }
            }
        }
        let mut annotated_files = FileWithAnnotatedLines::collect_annotations(self, args, msp);
        trace!("{annotated_files:#?}");
        let mut code_window_status = CodeWindowStatus::Open;

        // Make sure our primary file comes first
        let primary_span = msp.primary_span().unwrap_or_default();
        let (Some(sm), false) = (self.sm.as_ref(), primary_span.is_dummy()) else {
            // If we don't have span information, emit and exit
            return emit_to_destination(&buffer.render(), level, &mut self.dst, self.short_message)
                .map(|_| code_window_status);
        };
        let primary_lo = sm.lookup_char_pos(primary_span.lo());
        if let Ok(pos) =
            annotated_files.binary_search_by(|x| x.file.name.cmp(&primary_lo.file.name))
        {
            annotated_files.swap(0, pos);
        }

        // An end column separator should be emitted when a file with with a
        // source, is followed by one without a source
        let mut col_sep_before_no_show_source = false;
        let annotated_files_len = annotated_files.len();
        // Print out the annotate source lines that correspond with the error
        for (file_idx, annotated_file) in annotated_files.into_iter().enumerate() {
            // we can't annotate anything if the source is unavailable.
            if !should_show_source_code(
                &self.ignored_directories_in_source_blocks,
                sm,
                &annotated_file.file,
            ) {
                if !self.short_message {
                    // Add an end column separator when a file without a source
                    // comes after one with a source
                    //    ╭▸ $DIR/deriving-meta-unknown-trait.rs:1:10
                    //    │
                    // LL │ #[derive(Eqr)]
                    //    │          ━━━
                    //    ╰╴ (<- It prints *this* line)
                    //    ╭▸ $SRC_DIR/core/src/cmp.rs:356:0
                    //    │
                    //    ╰╴note: similarly named derive macro `Eq` defined here
                    if col_sep_before_no_show_source {
                        let buffer_msg_line_offset = buffer.num_lines();
                        self.draw_col_separator_end(
                            &mut buffer,
                            buffer_msg_line_offset,
                            max_line_num_len + 1,
                        );
                    }
                    col_sep_before_no_show_source = false;

                    // We'll just print an unannotated message.
                    for (annotation_id, line) in annotated_file.lines.iter().enumerate() {
                        let mut annotations = line.annotations.clone();
                        annotations.sort_by_key(|a| Reverse(a.start_col));
                        let mut line_idx = buffer.num_lines();

                        let labels: Vec<_> = annotations
                            .iter()
                            .filter_map(|a| Some((a.label.as_ref()?, a.is_primary)))
                            .filter(|(l, _)| !l.is_empty())
                            .collect();

                        if annotation_id == 0 || !labels.is_empty() {
                            buffer.append(
                                line_idx,
                                &format!(
                                    "{}:{}:{}",
                                    sm.filename_for_diagnostics(&annotated_file.file.name),
                                    sm.doctest_offset_line(
                                        &annotated_file.file.name,
                                        line.line_index
                                    ),
                                    annotations[0].start_col.file + 1,
                                ),
                                Style::LineAndColumn,
                            );
                            if annotation_id == 0 {
                                buffer.prepend(line_idx, self.file_start(), Style::LineNumber);
                            } else {
                                buffer.prepend(
                                    line_idx,
                                    self.secondary_file_start(),
                                    Style::LineNumber,
                                );
                            }
                            for _ in 0..max_line_num_len {
                                buffer.prepend(line_idx, " ", Style::NoStyle);
                            }
                            line_idx += 1;
                        }
                        if is_cont
                            && file_idx == annotated_files_len - 1
                            && annotation_id == annotated_file.lines.len() - 1
                            && !labels.is_empty()
                        {
                            code_window_status = CodeWindowStatus::Closed;
                        }
                        let labels_len = labels.len();
                        for (label_idx, (label, is_primary)) in labels.into_iter().enumerate() {
                            let style = if is_primary {
                                Style::LabelPrimary
                            } else {
                                Style::LabelSecondary
                            };
                            self.draw_col_separator_no_space(
                                &mut buffer,
                                line_idx,
                                max_line_num_len + 1,
                            );
                            line_idx += 1;
                            self.draw_note_separator(
                                &mut buffer,
                                line_idx,
                                max_line_num_len + 1,
                                label_idx != labels_len - 1,
                            );
                            buffer.append(line_idx, "note", Style::MainHeaderMsg);
                            buffer.append(line_idx, ": ", Style::NoStyle);
                            buffer.append(line_idx, label, style);
                            line_idx += 1;
                        }
                    }
                }
                continue;
            } else {
                col_sep_before_no_show_source = true;
            }

            // print out the span location and spacer before we print the annotated source
            // to do this, we need to know if this span will be primary
            let is_primary = primary_lo.file.name == annotated_file.file.name;
            if is_primary {
                let loc = primary_lo.clone();
                if !self.short_message {
                    // remember where we are in the output buffer for easy reference
                    let buffer_msg_line_offset = buffer.num_lines();

                    buffer.prepend(buffer_msg_line_offset, self.file_start(), Style::LineNumber);
                    buffer.append(
                        buffer_msg_line_offset,
                        &format!(
                            "{}:{}:{}",
                            sm.filename_for_diagnostics(&loc.file.name),
                            sm.doctest_offset_line(&loc.file.name, loc.line),
                            loc.col.0 + 1,
                        ),
                        Style::LineAndColumn,
                    );
                    for _ in 0..max_line_num_len {
                        buffer.prepend(buffer_msg_line_offset, " ", Style::NoStyle);
                    }
                } else {
                    buffer.prepend(
                        0,
                        &format!(
                            "{}:{}:{}: ",
                            sm.filename_for_diagnostics(&loc.file.name),
                            sm.doctest_offset_line(&loc.file.name, loc.line),
                            loc.col.0 + 1,
                        ),
                        Style::LineAndColumn,
                    );
                }
            } else if !self.short_message {
                // remember where we are in the output buffer for easy reference
                let buffer_msg_line_offset = buffer.num_lines();

                // Add spacing line, as shown:
                //   --> $DIR/file:54:15
                //    |
                // LL |         code
                //    |         ^^^^
                //    | (<- It prints *this* line)
                //   ::: $DIR/other_file.rs:15:5
                //    |
                // LL |     code
                //    |     ----
                self.draw_col_separator_no_space(
                    &mut buffer,
                    buffer_msg_line_offset,
                    max_line_num_len + 1,
                );

                // Then, the secondary file indicator
                buffer.prepend(
                    buffer_msg_line_offset + 1,
                    self.secondary_file_start(),
                    Style::LineNumber,
                );
                let loc = if let Some(first_line) = annotated_file.lines.first() {
                    let col = if let Some(first_annotation) = first_line.annotations.first() {
                        format!(":{}", first_annotation.start_col.file + 1)
                    } else {
                        String::new()
                    };
                    format!(
                        "{}:{}{}",
                        sm.filename_for_diagnostics(&annotated_file.file.name),
                        sm.doctest_offset_line(&annotated_file.file.name, first_line.line_index),
                        col
                    )
                } else {
                    format!("{}", sm.filename_for_diagnostics(&annotated_file.file.name))
                };
                buffer.append(buffer_msg_line_offset + 1, &loc, Style::LineAndColumn);
                for _ in 0..max_line_num_len {
                    buffer.prepend(buffer_msg_line_offset + 1, " ", Style::NoStyle);
                }
            }

            if !self.short_message {
                // Put in the spacer between the location and annotated source
                let buffer_msg_line_offset = buffer.num_lines();
                self.draw_col_separator_no_space(
                    &mut buffer,
                    buffer_msg_line_offset,
                    max_line_num_len + 1,
                );

                // Contains the vertical lines' positions for active multiline annotations
                let mut multilines = FxIndexMap::default();

                // Get the left-side margin to remove it
                let mut whitespace_margin = usize::MAX;
                for line_idx in 0..annotated_file.lines.len() {
                    let file = Arc::clone(&annotated_file.file);
                    let line = &annotated_file.lines[line_idx];
                    if let Some(source_string) =
                        line.line_index.checked_sub(1).and_then(|l| file.get_line(l))
                    {
                        // Whitespace can only be removed (aka considered leading)
                        // if the lexer considers it whitespace.
                        // non-rustc_lexer::is_whitespace() chars are reported as an
                        // error (ex. no-break-spaces \u{a0}), and thus can't be considered
                        // for removal during error reporting.
                        // FIXME: doesn't account for '\t' properly.
                        let leading_whitespace = source_string
                            .chars()
                            .take_while(|c| rustc_lexer::is_whitespace(*c))
                            .count();
                        if source_string.chars().any(|c| !rustc_lexer::is_whitespace(c)) {
                            whitespace_margin = min(whitespace_margin, leading_whitespace);
                        }
                    }
                }
                if whitespace_margin == usize::MAX {
                    whitespace_margin = 0;
                }

                // Left-most column any visible span points at.
                let mut span_left_margin = usize::MAX;
                for line in &annotated_file.lines {
                    for ann in &line.annotations {
                        span_left_margin = min(span_left_margin, ann.start_col.file);
                        span_left_margin = min(span_left_margin, ann.end_col.file);
                    }
                }
                if span_left_margin == usize::MAX {
                    span_left_margin = 0;
                }

                // Right-most column any visible span points at.
                let mut span_right_margin = 0;
                let mut label_right_margin = 0;
                let mut max_line_len = 0;
                for line in &annotated_file.lines {
                    max_line_len = max(
                        max_line_len,
                        line.line_index
                            .checked_sub(1)
                            .and_then(|l| annotated_file.file.get_line(l))
                            .map_or(0, |s| s.len()),
                    );
                    for ann in &line.annotations {
                        span_right_margin = max(span_right_margin, ann.start_col.file);
                        span_right_margin = max(span_right_margin, ann.end_col.file);
                        // FIXME: account for labels not in the same line
                        let label_right = ann.label.as_ref().map_or(0, |l| l.len() + 1);
                        label_right_margin =
                            max(label_right_margin, ann.end_col.file + label_right);
                    }
                }

                let width_offset = 3 + max_line_num_len;
                let code_offset = if annotated_file.multiline_depth == 0 {
                    width_offset
                } else {
                    width_offset + annotated_file.multiline_depth + 1
                };

                let column_width = self.column_width(code_offset);

                let margin = Margin::new(
                    whitespace_margin,
                    span_left_margin,
                    span_right_margin,
                    label_right_margin,
                    column_width,
                    max_line_len,
                );

                // Next, output the annotate source for this file
                for line_idx in 0..annotated_file.lines.len() {
                    let previous_buffer_line = buffer.num_lines();

                    let depths = self.render_source_line(
                        &mut buffer,
                        Arc::clone(&annotated_file.file),
                        &annotated_file.lines[line_idx],
                        width_offset,
                        code_offset,
                        margin,
                        !is_cont
                            && file_idx + 1 == annotated_files_len
                            && line_idx + 1 == annotated_file.lines.len(),
                    );

                    let mut to_add = FxIndexMap::default();

                    for (depth, style) in depths {
                        // FIXME(#120456) - is `swap_remove` correct?
                        if multilines.swap_remove(&depth).is_none() {
                            to_add.insert(depth, style);
                        }
                    }

                    // Set the multiline annotation vertical lines to the left of
                    // the code in this line.
                    for (depth, style) in &multilines {
                        for line in previous_buffer_line..buffer.num_lines() {
                            self.draw_multiline_line(
                                &mut buffer,
                                line,
                                width_offset,
                                *depth,
                                *style,
                            );
                        }
                    }
                    // check to see if we need to print out or elide lines that come between
                    // this annotated line and the next one.
                    if line_idx < (annotated_file.lines.len() - 1) {
                        let line_idx_delta = annotated_file.lines[line_idx + 1].line_index
                            - annotated_file.lines[line_idx].line_index;
                        if line_idx_delta > 2 {
                            let last_buffer_line_num = buffer.num_lines();
                            self.draw_line_separator(
                                &mut buffer,
                                last_buffer_line_num,
                                width_offset,
                            );

                            // Set the multiline annotation vertical lines on `...` bridging line.
                            for (depth, style) in &multilines {
                                self.draw_multiline_line(
                                    &mut buffer,
                                    last_buffer_line_num,
                                    width_offset,
                                    *depth,
                                    *style,
                                );
                            }
                            if let Some(line) = annotated_file.lines.get(line_idx) {
                                for ann in &line.annotations {
                                    if let AnnotationType::MultilineStart(pos) = ann.annotation_type
                                    {
                                        // In the case where we have elided the entire start of the
                                        // multispan because those lines were empty, we still need
                                        // to draw the `|`s across the `...`.
                                        self.draw_multiline_line(
                                            &mut buffer,
                                            last_buffer_line_num,
                                            width_offset,
                                            pos,
                                            if ann.is_primary {
                                                Style::UnderlinePrimary
                                            } else {
                                                Style::UnderlineSecondary
                                            },
                                        );
                                    }
                                }
                            }
                        } else if line_idx_delta == 2 {
                            let unannotated_line = annotated_file
                                .file
                                .get_line(annotated_file.lines[line_idx].line_index)
                                .unwrap_or_else(|| Cow::from(""));

                            let last_buffer_line_num = buffer.num_lines();

                            self.draw_line(
                                &mut buffer,
                                &normalize_whitespace(&unannotated_line),
                                annotated_file.lines[line_idx + 1].line_index - 1,
                                last_buffer_line_num,
                                width_offset,
                                code_offset,
                                margin,
                            );

                            for (depth, style) in &multilines {
                                self.draw_multiline_line(
                                    &mut buffer,
                                    last_buffer_line_num,
                                    width_offset,
                                    *depth,
                                    *style,
                                );
                            }
                            if let Some(line) = annotated_file.lines.get(line_idx) {
                                for ann in &line.annotations {
                                    if let AnnotationType::MultilineStart(pos) = ann.annotation_type
                                    {
                                        self.draw_multiline_line(
                                            &mut buffer,
                                            last_buffer_line_num,
                                            width_offset,
                                            pos,
                                            if ann.is_primary {
                                                Style::UnderlinePrimary
                                            } else {
                                                Style::UnderlineSecondary
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }

                    multilines.extend(&to_add);
                }
            }
            trace!("buffer: {:#?}", buffer.render());
        }

        // final step: take our styled buffer, render it, then output it
        emit_to_destination(&buffer.render(), level, &mut self.dst, self.short_message)?;

        Ok(code_window_status)
    }

    fn column_width(&self, code_offset: usize) -> usize {
        if let Some(width) = self.diagnostic_width {
            width.saturating_sub(code_offset)
        } else if self.ui_testing || cfg!(miri) {
            DEFAULT_COLUMN_WIDTH.saturating_sub(code_offset)
        } else {
            termize::dimensions()
                .map(|(w, _)| w.saturating_sub(code_offset))
                .unwrap_or(DEFAULT_COLUMN_WIDTH)
        }
    }

    fn emit_suggestion_default(
        &mut self,
        span: &MultiSpan,
        suggestion: &CodeSuggestion,
        args: &FluentArgs<'_>,
        level: &Level,
        max_line_num_len: usize,
    ) -> io::Result<()> {
        let Some(ref sm) = self.sm else {
            return Ok(());
        };

        // Render the replacements for each suggestion
        let suggestions = suggestion.splice_lines(sm);
        debug!(?suggestions);

        if suggestions.is_empty() {
            // Here we check if there are suggestions that have actual code changes. We sometimes
            // suggest the same code that is already there, instead of changing how we produce the
            // suggestions and filtering there, we just don't emit the suggestion.
            // Suggestions coming from macros can also have malformed spans. This is a heavy handed
            // approach to avoid ICEs by ignoring the suggestion outright.
            return Ok(());
        }

        let mut buffer = StyledBuffer::new();

        // Render the suggestion message
        buffer.append(0, level.to_str(), Style::Level(*level));
        buffer.append(0, ": ", Style::HeaderMsg);

        let mut msg = vec![(suggestion.msg.to_owned(), Style::NoStyle)];
        if let Some(confusion_type) =
            suggestions.iter().take(MAX_SUGGESTIONS).find_map(|(_, _, _, confusion_type)| {
                if confusion_type.has_confusion() { Some(*confusion_type) } else { None }
            })
        {
            msg.push((confusion_type.label_text().into(), Style::NoStyle));
        }
        self.msgs_to_buffer(
            &mut buffer,
            &msg,
            args,
            max_line_num_len,
            "suggestion",
            Some(Style::HeaderMsg),
        );

        let other_suggestions = suggestions.len().saturating_sub(MAX_SUGGESTIONS);

        let mut row_num = 2;
        for (i, (complete, parts, highlights, _)) in
            suggestions.into_iter().enumerate().take(MAX_SUGGESTIONS)
        {
            debug!(?complete, ?parts, ?highlights);

            let has_deletion =
                parts.iter().any(|p| p.is_deletion(sm) || p.is_destructive_replacement(sm));
            let is_multiline = complete.lines().count() > 1;

            if i == 0 {
                self.draw_col_separator_start(&mut buffer, row_num - 1, max_line_num_len + 1);
            } else {
                buffer.puts(
                    row_num - 1,
                    max_line_num_len + 1,
                    self.multi_suggestion_separator(),
                    Style::LineNumber,
                );
            }
            if let Some(span) = span.primary_span() {
                // Compare the primary span of the diagnostic with the span of the suggestion
                // being emitted. If they belong to the same file, we don't *need* to show the
                // file name, saving in verbosity, but if it *isn't* we do need it, otherwise we're
                // telling users to make a change but not clarifying *where*.
                let loc = sm.lookup_char_pos(parts[0].span.lo());
                if (span.is_dummy() || loc.file.name != sm.span_to_filename(span))
                    && loc.file.name.is_real()
                {
                    // --> file.rs:line:col
                    //  |
                    let arrow = self.file_start();
                    buffer.puts(row_num - 1, 0, arrow, Style::LineNumber);
                    let filename = sm.filename_for_diagnostics(&loc.file.name);
                    let offset = sm.doctest_offset_line(&loc.file.name, loc.line);
                    let message = format!("{}:{}:{}", filename, offset, loc.col.0 + 1);
                    if row_num == 2 {
                        let col = usize::max(max_line_num_len + 1, arrow.len());
                        buffer.puts(1, col, &message, Style::LineAndColumn);
                    } else {
                        buffer.append(row_num - 1, &message, Style::LineAndColumn);
                    }
                    for _ in 0..max_line_num_len {
                        buffer.prepend(row_num - 1, " ", Style::NoStyle);
                    }
                    self.draw_col_separator_no_space(&mut buffer, row_num, max_line_num_len + 1);
                    row_num += 1;
                }
            }
            let show_code_change = if has_deletion && !is_multiline {
                DisplaySuggestion::Diff
            } else if let [part] = &parts[..]
                && part.snippet.ends_with('\n')
                && part.snippet.trim() == complete.trim()
            {
                // We are adding a line(s) of code before code that was already there.
                DisplaySuggestion::Add
            } else if (parts.len() != 1 || parts[0].snippet.trim() != complete.trim())
                && !is_multiline
            {
                DisplaySuggestion::Underline
            } else {
                DisplaySuggestion::None
            };

            if let DisplaySuggestion::Diff = show_code_change {
                row_num += 1;
            }

            let file_lines = sm
                .span_to_lines(parts[0].span)
                .expect("span_to_lines failed when emitting suggestion");

            assert!(!file_lines.lines.is_empty() || parts[0].span.is_dummy());

            let line_start = sm.lookup_char_pos(parts[0].span.lo()).line;
            let mut lines = complete.lines();
            if lines.clone().next().is_none() {
                // Account for a suggestion to completely remove a line(s) with whitespace (#94192).
                let line_end = sm.lookup_char_pos(parts[0].span.hi()).line;
                for line in line_start..=line_end {
                    self.draw_line_num(
                        &mut buffer,
                        line,
                        row_num - 1 + line - line_start,
                        max_line_num_len,
                    );
                    buffer.puts(
                        row_num - 1 + line - line_start,
                        max_line_num_len + 1,
                        "- ",
                        Style::Removal,
                    );
                    buffer.puts(
                        row_num - 1 + line - line_start,
                        max_line_num_len + 3,
                        &normalize_whitespace(&file_lines.file.get_line(line - 1).unwrap()),
                        Style::Removal,
                    );
                }
                row_num += line_end - line_start;
            }
            let mut unhighlighted_lines = Vec::new();
            let mut last_pos = 0;
            let mut is_item_attribute = false;
            for (line_pos, (line, highlight_parts)) in lines.by_ref().zip(highlights).enumerate() {
                last_pos = line_pos;
                debug!(%line_pos, %line, ?highlight_parts);

                // Remember lines that are not highlighted to hide them if needed
                if highlight_parts.is_empty() {
                    unhighlighted_lines.push((line_pos, line));
                    continue;
                }
                if highlight_parts.len() == 1
                    && line.trim().starts_with("#[")
                    && line.trim().ends_with(']')
                {
                    is_item_attribute = true;
                }

                match unhighlighted_lines.len() {
                    0 => (),
                    // Since we show first line, "..." line and last line,
                    // There is no reason to hide if there are 3 or less lines
                    // (because then we just replace a line with ... which is
                    // not helpful)
                    n if n <= 3 => unhighlighted_lines.drain(..).for_each(|(p, l)| {
                        self.draw_code_line(
                            &mut buffer,
                            &mut row_num,
                            &[],
                            p + line_start,
                            l,
                            show_code_change,
                            max_line_num_len,
                            &file_lines,
                            is_multiline,
                        )
                    }),
                    // Print first unhighlighted line, "..." and last unhighlighted line, like so:
                    //
                    // LL | this line was highlighted
                    // LL | this line is just for context
                    // ...
                    // LL | this line is just for context
                    // LL | this line was highlighted
                    _ => {
                        let last_line = unhighlighted_lines.pop();
                        let first_line = unhighlighted_lines.drain(..).next();

                        if let Some((p, l)) = first_line {
                            self.draw_code_line(
                                &mut buffer,
                                &mut row_num,
                                &[],
                                p + line_start,
                                l,
                                show_code_change,
                                max_line_num_len,
                                &file_lines,
                                is_multiline,
                            )
                        }

                        let placeholder = self.margin();
                        let padding = str_width(placeholder);
                        buffer.puts(
                            row_num,
                            max_line_num_len.saturating_sub(padding),
                            placeholder,
                            Style::LineNumber,
                        );
                        row_num += 1;

                        if let Some((p, l)) = last_line {
                            self.draw_code_line(
                                &mut buffer,
                                &mut row_num,
                                &[],
                                p + line_start,
                                l,
                                show_code_change,
                                max_line_num_len,
                                &file_lines,
                                is_multiline,
                            )
                        }
                    }
                }

                self.draw_code_line(
                    &mut buffer,
                    &mut row_num,
                    &highlight_parts,
                    line_pos + line_start,
                    line,
                    show_code_change,
                    max_line_num_len,
                    &file_lines,
                    is_multiline,
                )
            }
            if let DisplaySuggestion::Add = show_code_change
                && is_item_attribute
            {
                // The suggestion adds an entire line of code, ending on a newline, so we'll also
                // print the *following* line, to provide context of what we're advising people to
                // do. Otherwise you would only see contextless code that can be confused for
                // already existing code, despite the colors and UI elements.
                // We special case `#[derive(_)]\n` and other attribute suggestions, because those
                // are the ones where context is most useful.
                let file_lines = sm
                    .span_to_lines(parts[0].span.shrink_to_hi())
                    .expect("span_to_lines failed when emitting suggestion");
                let line_num = sm.lookup_char_pos(parts[0].span.lo()).line;
                if let Some(line) = file_lines.file.get_line(line_num - 1) {
                    let line = normalize_whitespace(&line);
                    self.draw_code_line(
                        &mut buffer,
                        &mut row_num,
                        &[],
                        line_num + last_pos + 1,
                        &line,
                        DisplaySuggestion::None,
                        max_line_num_len,
                        &file_lines,
                        is_multiline,
                    )
                }
            }

            // This offset and the ones below need to be signed to account for replacement code
            // that is shorter than the original code.
            let mut offsets: Vec<(usize, isize)> = Vec::new();
            // Only show an underline in the suggestions if the suggestion is not the
            // entirety of the code being shown and the displayed code is not multiline.
            if let DisplaySuggestion::Diff | DisplaySuggestion::Underline | DisplaySuggestion::Add =
                show_code_change
            {
                for part in parts {
                    let snippet = if let Ok(snippet) = sm.span_to_snippet(part.span) {
                        snippet
                    } else {
                        String::new()
                    };
                    let span_start_pos = sm.lookup_char_pos(part.span.lo()).col_display;
                    let span_end_pos = sm.lookup_char_pos(part.span.hi()).col_display;

                    // If this addition is _only_ whitespace, then don't trim it,
                    // or else we're just not rendering anything.
                    let is_whitespace_addition = part.snippet.trim().is_empty();

                    // Do not underline the leading...
                    let start = if is_whitespace_addition {
                        0
                    } else {
                        part.snippet.len().saturating_sub(part.snippet.trim_start().len())
                    };
                    // ...or trailing spaces. Account for substitutions containing unicode
                    // characters.
                    let sub_len: usize = str_width(if is_whitespace_addition {
                        &part.snippet
                    } else {
                        part.snippet.trim()
                    });

                    let offset: isize = offsets
                        .iter()
                        .filter_map(
                            |(start, v)| if span_start_pos < *start { None } else { Some(v) },
                        )
                        .sum();
                    let underline_start = (span_start_pos + start) as isize + offset;
                    let underline_end = (span_start_pos + start + sub_len) as isize + offset;
                    let padding: usize = max_line_num_len + 3;
                    for p in underline_start..underline_end {
                        if let DisplaySuggestion::Underline = show_code_change
                            && is_different(sm, &part.snippet, part.span)
                        {
                            // If this is a replacement, underline with `~`, if this is an addition
                            // underline with `+`.
                            buffer.putc(
                                row_num,
                                (padding as isize + p) as usize,
                                if part.is_addition(sm) { '+' } else { self.diff() },
                                Style::Addition,
                            );
                        }
                    }
                    if let DisplaySuggestion::Diff = show_code_change {
                        // Colorize removal with red in diff format.

                        // Below, there's some tricky buffer indexing going on. `row_num` at this
                        // point corresponds to:
                        //
                        //    |
                        // LL | CODE
                        //    | ++++  <- `row_num`
                        //
                        // in the buffer. When we have a diff format output, we end up with
                        //
                        //    |
                        // LL - OLDER   <- row_num - 2
                        // LL + NEWER
                        //    |         <- row_num
                        //
                        // The `row_num - 2` is to select the buffer line that has the "old version
                        // of the diff" at that point. When the removal is a single line, `i` is
                        // `0`, `newlines` is `1` so `(newlines - i - 1)` ends up being `0`, so row
                        // points at `LL - OLDER`. When the removal corresponds to multiple lines,
                        // we end up with `newlines > 1` and `i` being `0..newlines - 1`.
                        //
                        //    |
                        // LL - OLDER   <- row_num - 2 - (newlines - last_i - 1)
                        // LL - CODE
                        // LL - BEING
                        // LL - REMOVED <- row_num - 2 - (newlines - first_i - 1)
                        // LL + NEWER
                        //    |         <- row_num

                        let newlines = snippet.lines().count();
                        if newlines > 0 && row_num > newlines {
                            // Account for removals where the part being removed spans multiple
                            // lines.
                            // FIXME: We check the number of rows because in some cases, like in
                            // `tests/ui/lint/invalid-nan-comparison-suggestion.rs`, the rendered
                            // suggestion will only show the first line of code being replaced. The
                            // proper way of doing this would be to change the suggestion rendering
                            // logic to show the whole prior snippet, but the current output is not
                            // too bad to begin with, so we side-step that issue here.
                            for (i, line) in snippet.lines().enumerate() {
                                let line = normalize_whitespace(line);
                                let row = (row_num - 2 - (newlines - i - 1)).max(2);
                                // On the first line, we highlight between the start of the part
                                // span, and the end of that line.
                                // On the last line, we highlight between the start of the line, and
                                // the column of the part span end.
                                // On all others, we highlight the whole line.
                                let start = if i == 0 {
                                    (padding as isize + span_start_pos as isize) as usize
                                } else {
                                    padding
                                };
                                let end = if i == 0 {
                                    (padding as isize
                                        + span_start_pos as isize
                                        + line.len() as isize)
                                        as usize
                                } else if i == newlines - 1 {
                                    (padding as isize + span_end_pos as isize) as usize
                                } else {
                                    (padding as isize + line.len() as isize) as usize
                                };
                                buffer.set_style_range(row, start, end, Style::Removal, true);
                            }
                        } else {
                            // The removed code fits all in one line.
                            buffer.set_style_range(
                                row_num - 2,
                                (padding as isize + span_start_pos as isize) as usize,
                                (padding as isize + span_end_pos as isize) as usize,
                                Style::Removal,
                                true,
                            );
                        }
                    }

                    // length of the code after substitution
                    let full_sub_len = str_width(&part.snippet) as isize;

                    // length of the code to be substituted
                    let snippet_len = span_end_pos as isize - span_start_pos as isize;
                    // For multiple substitutions, use the position *after* the previous
                    // substitutions have happened, only when further substitutions are
                    // located strictly after.
                    offsets.push((span_end_pos, full_sub_len - snippet_len));
                }
                row_num += 1;
            }

            // if we elided some lines, add an ellipsis
            if lines.next().is_some() {
                let placeholder = self.margin();
                let padding = str_width(placeholder);
                buffer.puts(
                    row_num,
                    max_line_num_len.saturating_sub(padding),
                    placeholder,
                    Style::LineNumber,
                );
            } else {
                let row = match show_code_change {
                    DisplaySuggestion::Diff
                    | DisplaySuggestion::Add
                    | DisplaySuggestion::Underline => row_num - 1,
                    DisplaySuggestion::None => row_num,
                };
                if other_suggestions > 0 {
                    self.draw_col_separator_no_space(&mut buffer, row, max_line_num_len + 1);
                } else {
                    self.draw_col_separator_end(&mut buffer, row, max_line_num_len + 1);
                }
                row_num = row + 1;
            }
        }
        if other_suggestions > 0 {
            self.draw_note_separator(&mut buffer, row_num, max_line_num_len + 1, false);
            let msg = format!(
                "and {} other candidate{}",
                other_suggestions,
                pluralize!(other_suggestions)
            );
            buffer.append(row_num, &msg, Style::NoStyle);
        }

        emit_to_destination(&buffer.render(), level, &mut self.dst, self.short_message)?;
        Ok(())
    }

    #[instrument(level = "trace", skip(self, args, code, children, suggestions))]
    fn emit_messages_default(
        &mut self,
        level: &Level,
        messages: &[(DiagMessage, Style)],
        args: &FluentArgs<'_>,
        code: &Option<ErrCode>,
        span: &MultiSpan,
        children: &[Subdiag],
        suggestions: &[CodeSuggestion],
    ) {
        let max_line_num_len = if self.ui_testing {
            ANONYMIZED_LINE_NUM.len()
        } else {
            let n = self.get_max_line_num(span, children);
            num_decimal_digits(n)
        };

        match self.emit_messages_default_inner(
            span,
            messages,
            args,
            code,
            level,
            max_line_num_len,
            false,
            !children.is_empty()
                || suggestions.iter().any(|s| s.style != SuggestionStyle::CompletelyHidden),
        ) {
            Ok(code_window_status) => {
                if !children.is_empty()
                    || suggestions.iter().any(|s| s.style != SuggestionStyle::CompletelyHidden)
                {
                    let mut buffer = StyledBuffer::new();
                    if !self.short_message {
                        if let Some(child) = children.iter().next()
                            && child.span.primary_spans().is_empty()
                        {
                            // We'll continue the vertical bar to point into the next note.
                            self.draw_col_separator_no_space(&mut buffer, 0, max_line_num_len + 1);
                        } else if matches!(code_window_status, CodeWindowStatus::Open) {
                            // We'll close the vertical bar to visually end the code window.
                            self.draw_col_separator_end(&mut buffer, 0, max_line_num_len + 1);
                        }
                    }
                    if let Err(e) = emit_to_destination(
                        &buffer.render(),
                        level,
                        &mut self.dst,
                        self.short_message,
                    ) {
                        panic!("failed to emit error: {e}")
                    }
                }
                if !self.short_message {
                    for (i, child) in children.iter().enumerate() {
                        assert!(child.level.can_be_subdiag());
                        let span = &child.span;
                        // FIXME: audit that this behaves correctly with suggestions.
                        let should_close = match children.get(i + 1) {
                            Some(c) => !c.span.primary_spans().is_empty(),
                            None => i + 1 == children.len(),
                        };
                        if let Err(err) = self.emit_messages_default_inner(
                            span,
                            &child.messages,
                            args,
                            &None,
                            &child.level,
                            max_line_num_len,
                            true,
                            !should_close,
                        ) {
                            panic!("failed to emit error: {err}");
                        }
                    }
                    for (i, sugg) in suggestions.iter().enumerate() {
                        match sugg.style {
                            SuggestionStyle::CompletelyHidden => {
                                // do not display this suggestion, it is meant only for tools
                            }
                            SuggestionStyle::HideCodeAlways => {
                                if let Err(e) = self.emit_messages_default_inner(
                                    &MultiSpan::new(),
                                    &[(sugg.msg.to_owned(), Style::HeaderMsg)],
                                    args,
                                    &None,
                                    &Level::Help,
                                    max_line_num_len,
                                    true,
                                    // FIXME: this needs to account for the suggestion type,
                                    //        some don't take any space.
                                    i + 1 != suggestions.len(),
                                ) {
                                    panic!("failed to emit error: {e}");
                                }
                            }
                            SuggestionStyle::HideCodeInline
                            | SuggestionStyle::ShowCode
                            | SuggestionStyle::ShowAlways => {
                                if let Err(e) = self.emit_suggestion_default(
                                    span,
                                    sugg,
                                    args,
                                    &Level::Help,
                                    max_line_num_len,
                                ) {
                                    panic!("failed to emit error: {e}");
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => panic!("failed to emit error: {e}"),
        }

        match writeln!(self.dst) {
            Err(e) => panic!("failed to emit error: {e}"),
            _ => {
                if let Err(e) = self.dst.flush() {
                    panic!("failed to emit error: {e}")
                }
            }
        }
    }

    fn draw_code_line(
        &self,
        buffer: &mut StyledBuffer,
        row_num: &mut usize,
        highlight_parts: &[SubstitutionHighlight],
        line_num: usize,
        line_to_add: &str,
        show_code_change: DisplaySuggestion,
        max_line_num_len: usize,
        file_lines: &FileLines,
        is_multiline: bool,
    ) {
        if let DisplaySuggestion::Diff = show_code_change {
            // We need to print more than one line if the span we need to remove is multiline.
            // For more info: https://github.com/rust-lang/rust/issues/92741
            let lines_to_remove = file_lines.lines.iter().take(file_lines.lines.len() - 1);
            for (index, line_to_remove) in lines_to_remove.enumerate() {
                self.draw_line_num(buffer, line_num + index, *row_num - 1, max_line_num_len);
                buffer.puts(*row_num - 1, max_line_num_len + 1, "- ", Style::Removal);
                let line = normalize_whitespace(
                    &file_lines.file.get_line(line_to_remove.line_index).unwrap(),
                );
                buffer.puts(*row_num - 1, max_line_num_len + 3, &line, Style::NoStyle);
                *row_num += 1;
            }
            // If the last line is exactly equal to the line we need to add, we can skip both of
            // them. This allows us to avoid output like the following:
            // 2 - &
            // 2 + if true { true } else { false }
            // 3 - if true { true } else { false }
            // If those lines aren't equal, we print their diff
            let last_line_index = file_lines.lines[file_lines.lines.len() - 1].line_index;
            let last_line = &file_lines.file.get_line(last_line_index).unwrap();
            if last_line != line_to_add {
                self.draw_line_num(
                    buffer,
                    line_num + file_lines.lines.len() - 1,
                    *row_num - 1,
                    max_line_num_len,
                );
                buffer.puts(*row_num - 1, max_line_num_len + 1, "- ", Style::Removal);
                buffer.puts(
                    *row_num - 1,
                    max_line_num_len + 3,
                    &normalize_whitespace(last_line),
                    Style::NoStyle,
                );
                if !line_to_add.trim().is_empty() {
                    // Check if after the removal, the line is left with only whitespace. If so, we
                    // will not show an "addition" line, as removing the whole line is what the user
                    // would really want.
                    // For example, for the following:
                    //   |
                    // 2 -     .await
                    // 2 +     (note the left over whitespace)
                    //   |
                    // We really want
                    //   |
                    // 2 -     .await
                    //   |
                    // *row_num -= 1;
                    self.draw_line_num(buffer, line_num, *row_num, max_line_num_len);
                    buffer.puts(*row_num, max_line_num_len + 1, "+ ", Style::Addition);
                    buffer.append(*row_num, &normalize_whitespace(line_to_add), Style::NoStyle);
                } else {
                    *row_num -= 1;
                }
            } else {
                *row_num -= 2;
            }
        } else if is_multiline {
            self.draw_line_num(buffer, line_num, *row_num, max_line_num_len);
            match &highlight_parts {
                [SubstitutionHighlight { start: 0, end }] if *end == line_to_add.len() => {
                    buffer.puts(*row_num, max_line_num_len + 1, "+ ", Style::Addition);
                }
                [] => {
                    // FIXME: needed? Doesn't get exercised in any test.
                    self.draw_col_separator_no_space(buffer, *row_num, max_line_num_len + 1);
                }
                _ => {
                    let diff = self.diff();
                    buffer.puts(
                        *row_num,
                        max_line_num_len + 1,
                        &format!("{diff} "),
                        Style::Addition,
                    );
                }
            }
            //   LL | line_to_add
            //   ++^^^
            //    |  |
            //    |  magic `3`
            //    `max_line_num_len`
            buffer.puts(
                *row_num,
                max_line_num_len + 3,
                &normalize_whitespace(line_to_add),
                Style::NoStyle,
            );
        } else if let DisplaySuggestion::Add = show_code_change {
            self.draw_line_num(buffer, line_num, *row_num, max_line_num_len);
            buffer.puts(*row_num, max_line_num_len + 1, "+ ", Style::Addition);
            buffer.append(*row_num, &normalize_whitespace(line_to_add), Style::NoStyle);
        } else {
            self.draw_line_num(buffer, line_num, *row_num, max_line_num_len);
            self.draw_col_separator(buffer, *row_num, max_line_num_len + 1);
            buffer.append(*row_num, &normalize_whitespace(line_to_add), Style::NoStyle);
        }

        // Colorize addition/replacements with green.
        for &SubstitutionHighlight { start, end } in highlight_parts {
            // This is a no-op for empty ranges
            if start != end {
                // Account for tabs when highlighting (#87972).
                let tabs: usize = line_to_add
                    .chars()
                    .take(start)
                    .map(|ch| match ch {
                        '\t' => 3,
                        _ => 0,
                    })
                    .sum();
                buffer.set_style_range(
                    *row_num,
                    max_line_num_len + 3 + start + tabs,
                    max_line_num_len + 3 + end + tabs,
                    Style::Addition,
                    true,
                );
            }
        }
        *row_num += 1;
    }

    fn underline(&self, is_primary: bool) -> UnderlineParts {
        //               X0 Y0
        // label_start > ┯━━━━ < underline
        //               │ < vertical_text_line
        //               text

        //    multiline_start_down ⤷ X0 Y0
        //            top_left > ┌───╿──┘ < top_right_flat
        //           top_left > ┏│━━━┙ < top_right
        // multiline_vertical > ┃│
        //                      ┃│   X1 Y1
        //                      ┃│   X2 Y2
        //                      ┃└────╿──┘ < multiline_end_same_line
        //        bottom_left > ┗━━━━━┥ < bottom_right_with_text
        //   multiline_horizontal ^   `X` is a good letter

        // multiline_whole_line > ┏ X0 Y0
        //                        ┃   X1 Y1
        //                        ┗━━━━┛ < multiline_end_same_line

        // multiline_whole_line > ┏ X0 Y0
        //                        ┃ X1 Y1
        //                        ┃  ╿ < multiline_end_up
        //                        ┗━━┛ < bottom_right

        match (self.theme, is_primary) {
            (OutputTheme::Ascii, true) => UnderlineParts {
                style: Style::UnderlinePrimary,
                underline: '^',
                label_start: '^',
                vertical_text_line: '|',
                multiline_vertical: '|',
                multiline_horizontal: '_',
                multiline_whole_line: '/',
                multiline_start_down: '^',
                bottom_right: '|',
                top_left: ' ',
                top_right_flat: '^',
                bottom_left: '|',
                multiline_end_up: '^',
                multiline_end_same_line: '^',
                multiline_bottom_right_with_text: '|',
            },
            (OutputTheme::Ascii, false) => UnderlineParts {
                style: Style::UnderlineSecondary,
                underline: '-',
                label_start: '-',
                vertical_text_line: '|',
                multiline_vertical: '|',
                multiline_horizontal: '_',
                multiline_whole_line: '/',
                multiline_start_down: '-',
                bottom_right: '|',
                top_left: ' ',
                top_right_flat: '-',
                bottom_left: '|',
                multiline_end_up: '-',
                multiline_end_same_line: '-',
                multiline_bottom_right_with_text: '|',
            },
            (OutputTheme::Unicode, true) => UnderlineParts {
                style: Style::UnderlinePrimary,
                underline: '━',
                label_start: '┯',
                vertical_text_line: '│',
                multiline_vertical: '┃',
                multiline_horizontal: '━',
                multiline_whole_line: '┏',
                multiline_start_down: '╿',
                bottom_right: '┙',
                top_left: '┏',
                top_right_flat: '┛',
                bottom_left: '┗',
                multiline_end_up: '╿',
                multiline_end_same_line: '┛',
                multiline_bottom_right_with_text: '┥',
            },
            (OutputTheme::Unicode, false) => UnderlineParts {
                style: Style::UnderlineSecondary,
                underline: '─',
                label_start: '┬',
                vertical_text_line: '│',
                multiline_vertical: '│',
                multiline_horizontal: '─',
                multiline_whole_line: '┌',
                multiline_start_down: '│',
                bottom_right: '┘',
                top_left: '┌',
                top_right_flat: '┘',
                bottom_left: '└',
                multiline_end_up: '│',
                multiline_end_same_line: '┘',
                multiline_bottom_right_with_text: '┤',
            },
        }
    }

    fn col_separator(&self) -> char {
        match self.theme {
            OutputTheme::Ascii => '|',
            OutputTheme::Unicode => '│',
        }
    }

    fn note_separator(&self, is_cont: bool) -> &'static str {
        match self.theme {
            OutputTheme::Ascii => "= ",
            OutputTheme::Unicode if is_cont => "├ ",
            OutputTheme::Unicode => "╰ ",
        }
    }

    fn multi_suggestion_separator(&self) -> &'static str {
        match self.theme {
            OutputTheme::Ascii => "|",
            OutputTheme::Unicode => "├╴",
        }
    }

    fn draw_col_separator(&self, buffer: &mut StyledBuffer, line: usize, col: usize) {
        let chr = self.col_separator();
        buffer.puts(line, col, &format!("{chr} "), Style::LineNumber);
    }

    fn draw_col_separator_no_space(&self, buffer: &mut StyledBuffer, line: usize, col: usize) {
        let chr = self.col_separator();
        self.draw_col_separator_no_space_with_style(buffer, chr, line, col, Style::LineNumber);
    }

    fn draw_col_separator_start(&self, buffer: &mut StyledBuffer, line: usize, col: usize) {
        match self.theme {
            OutputTheme::Ascii => {
                self.draw_col_separator_no_space_with_style(
                    buffer,
                    '|',
                    line,
                    col,
                    Style::LineNumber,
                );
            }
            OutputTheme::Unicode => {
                self.draw_col_separator_no_space_with_style(
                    buffer,
                    '╭',
                    line,
                    col,
                    Style::LineNumber,
                );
                self.draw_col_separator_no_space_with_style(
                    buffer,
                    '╴',
                    line,
                    col + 1,
                    Style::LineNumber,
                );
            }
        }
    }

    fn draw_col_separator_end(&self, buffer: &mut StyledBuffer, line: usize, col: usize) {
        match self.theme {
            OutputTheme::Ascii => {
                self.draw_col_separator_no_space_with_style(
                    buffer,
                    '|',
                    line,
                    col,
                    Style::LineNumber,
                );
            }
            OutputTheme::Unicode => {
                self.draw_col_separator_no_space_with_style(
                    buffer,
                    '╰',
                    line,
                    col,
                    Style::LineNumber,
                );
                self.draw_col_separator_no_space_with_style(
                    buffer,
                    '╴',
                    line,
                    col + 1,
                    Style::LineNumber,
                );
            }
        }
    }

    fn draw_col_separator_no_space_with_style(
        &self,
        buffer: &mut StyledBuffer,
        chr: char,
        line: usize,
        col: usize,
        style: Style,
    ) {
        buffer.putc(line, col, chr, style);
    }

    fn draw_range(
        &self,
        buffer: &mut StyledBuffer,
        symbol: char,
        line: usize,
        col_from: usize,
        col_to: usize,
        style: Style,
    ) {
        for col in col_from..col_to {
            buffer.putc(line, col, symbol, style);
        }
    }

    fn draw_note_separator(
        &self,
        buffer: &mut StyledBuffer,
        line: usize,
        col: usize,
        is_cont: bool,
    ) {
        let chr = self.note_separator(is_cont);
        buffer.puts(line, col, chr, Style::LineNumber);
    }

    fn draw_multiline_line(
        &self,
        buffer: &mut StyledBuffer,
        line: usize,
        offset: usize,
        depth: usize,
        style: Style,
    ) {
        let chr = match (style, self.theme) {
            (Style::UnderlinePrimary | Style::LabelPrimary, OutputTheme::Ascii) => '|',
            (_, OutputTheme::Ascii) => '|',
            (Style::UnderlinePrimary | Style::LabelPrimary, OutputTheme::Unicode) => '┃',
            (_, OutputTheme::Unicode) => '│',
        };
        buffer.putc(line, offset + depth - 1, chr, style);
    }

    fn file_start(&self) -> &'static str {
        match self.theme {
            OutputTheme::Ascii => "--> ",
            OutputTheme::Unicode => " ╭▸ ",
        }
    }

    fn secondary_file_start(&self) -> &'static str {
        match self.theme {
            OutputTheme::Ascii => "::: ",
            OutputTheme::Unicode => " ⸬  ",
        }
    }

    fn diff(&self) -> char {
        match self.theme {
            OutputTheme::Ascii => '~',
            OutputTheme::Unicode => '±',
        }
    }

    fn draw_line_separator(&self, buffer: &mut StyledBuffer, line: usize, col: usize) {
        let (column, dots) = match self.theme {
            OutputTheme::Ascii => (0, "..."),
            OutputTheme::Unicode => (col - 2, "‡"),
        };
        buffer.puts(line, column, dots, Style::LineNumber);
    }

    fn margin(&self) -> &'static str {
        match self.theme {
            OutputTheme::Ascii => "...",
            OutputTheme::Unicode => "…",
        }
    }

    fn draw_line_num(
        &self,
        buffer: &mut StyledBuffer,
        line_num: usize,
        line_offset: usize,
        max_line_num_len: usize,
    ) {
        let line_num = self.maybe_anonymized(line_num);
        buffer.puts(
            line_offset,
            max_line_num_len.saturating_sub(str_width(&line_num)),
            &line_num,
            Style::LineNumber,
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct UnderlineParts {
    style: Style,
    underline: char,
    label_start: char,
    vertical_text_line: char,
    multiline_vertical: char,
    multiline_horizontal: char,
    multiline_whole_line: char,
    multiline_start_down: char,
    bottom_right: char,
    top_left: char,
    top_right_flat: char,
    bottom_left: char,
    multiline_end_up: char,
    multiline_end_same_line: char,
    multiline_bottom_right_with_text: char,
}

#[derive(Clone, Copy, Debug)]
enum DisplaySuggestion {
    Underline,
    Diff,
    None,
    Add,
}

#[derive(Clone, Copy, Debug)]
enum CodeWindowStatus {
    Closed,
    Open,
}

impl FileWithAnnotatedLines {
    /// Preprocess all the annotations so that they are grouped by file and by line number
    /// This helps us quickly iterate over the whole message (including secondary file spans)
    pub(crate) fn collect_annotations(
        emitter: &dyn Emitter,
        args: &FluentArgs<'_>,
        msp: &MultiSpan,
    ) -> Vec<FileWithAnnotatedLines> {
        fn add_annotation_to_file(
            file_vec: &mut Vec<FileWithAnnotatedLines>,
            file: Arc<SourceFile>,
            line_index: usize,
            ann: Annotation,
        ) {
            for slot in file_vec.iter_mut() {
                // Look through each of our files for the one we're adding to
                if slot.file.name == file.name {
                    // See if we already have a line for it
                    for line_slot in &mut slot.lines {
                        if line_slot.line_index == line_index {
                            line_slot.annotations.push(ann);
                            return;
                        }
                    }
                    // We don't have a line yet, create one
                    slot.lines.push(Line { line_index, annotations: vec![ann] });
                    slot.lines.sort();
                    return;
                }
            }
            // This is the first time we're seeing the file
            file_vec.push(FileWithAnnotatedLines {
                file,
                lines: vec![Line { line_index, annotations: vec![ann] }],
                multiline_depth: 0,
            });
        }

        let mut output = vec![];
        let mut multiline_annotations = vec![];

        if let Some(sm) = emitter.source_map() {
            for SpanLabel { span, is_primary, label } in msp.span_labels() {
                // If we don't have a useful span, pick the primary span if that exists.
                // Worst case we'll just print an error at the top of the main file.
                let span = match (span.is_dummy(), msp.primary_span()) {
                    (_, None) | (false, _) => span,
                    (true, Some(span)) => span,
                };

                let lo = sm.lookup_char_pos(span.lo());
                let mut hi = sm.lookup_char_pos(span.hi());

                // Watch out for "empty spans". If we get a span like 6..6, we
                // want to just display a `^` at 6, so convert that to
                // 6..7. This is degenerate input, but it's best to degrade
                // gracefully -- and the parser likes to supply a span like
                // that for EOF, in particular.

                if lo.col_display == hi.col_display && lo.line == hi.line {
                    hi.col_display += 1;
                }

                let label = label.as_ref().map(|m| {
                    normalize_whitespace(
                        &emitter
                            .translator()
                            .translate_message(m, args)
                            .map_err(Report::new)
                            .unwrap(),
                    )
                });

                if lo.line != hi.line {
                    let ml = MultilineAnnotation {
                        depth: 1,
                        line_start: lo.line,
                        line_end: hi.line,
                        start_col: AnnotationColumn::from_loc(&lo),
                        end_col: AnnotationColumn::from_loc(&hi),
                        is_primary,
                        label,
                        overlaps_exactly: false,
                    };
                    multiline_annotations.push((lo.file, ml));
                } else {
                    let ann = Annotation {
                        start_col: AnnotationColumn::from_loc(&lo),
                        end_col: AnnotationColumn::from_loc(&hi),
                        is_primary,
                        label,
                        annotation_type: AnnotationType::Singleline,
                    };
                    add_annotation_to_file(&mut output, lo.file, lo.line, ann);
                };
            }
        }

        // Find overlapping multiline annotations, put them at different depths
        multiline_annotations.sort_by_key(|(_, ml)| (ml.line_start, usize::MAX - ml.line_end));
        for (_, ann) in multiline_annotations.clone() {
            for (_, a) in multiline_annotations.iter_mut() {
                // Move all other multiline annotations overlapping with this one
                // one level to the right.
                if !(ann.same_span(a))
                    && num_overlap(ann.line_start, ann.line_end, a.line_start, a.line_end, true)
                {
                    a.increase_depth();
                } else if ann.same_span(a) && &ann != a {
                    a.overlaps_exactly = true;
                } else {
                    break;
                }
            }
        }

        let mut max_depth = 0; // max overlapping multiline spans
        for (_, ann) in &multiline_annotations {
            max_depth = max(max_depth, ann.depth);
        }
        // Change order of multispan depth to minimize the number of overlaps in the ASCII art.
        for (_, a) in multiline_annotations.iter_mut() {
            a.depth = max_depth - a.depth + 1;
        }
        for (file, ann) in multiline_annotations {
            let mut end_ann = ann.as_end();
            if !ann.overlaps_exactly {
                // avoid output like
                //
                //  |        foo(
                //  |   _____^
                //  |  |_____|
                //  | ||         bar,
                //  | ||     );
                //  | ||      ^
                //  | ||______|
                //  |  |______foo
                //  |         baz
                //
                // and instead get
                //
                //  |       foo(
                //  |  _____^
                //  | |         bar,
                //  | |     );
                //  | |      ^
                //  | |      |
                //  | |______foo
                //  |        baz
                add_annotation_to_file(
                    &mut output,
                    Arc::clone(&file),
                    ann.line_start,
                    ann.as_start(),
                );
                // 4 is the minimum vertical length of a multiline span when presented: two lines
                // of code and two lines of underline. This is not true for the special case where
                // the beginning doesn't have an underline, but the current logic seems to be
                // working correctly.
                let middle = min(ann.line_start + 4, ann.line_end);
                // We'll show up to 4 lines past the beginning of the multispan start.
                // We will *not* include the tail of lines that are only whitespace, a comment or
                // a bare delimiter.
                let filter = |s: &str| {
                    let s = s.trim();
                    // Consider comments as empty, but don't consider docstrings to be empty.
                    !(s.starts_with("//") && !(s.starts_with("///") || s.starts_with("//!")))
                        // Consider lines with nothing but whitespace, a single delimiter as empty.
                        && !["", "{", "}", "(", ")", "[", "]"].contains(&s)
                };
                let until = (ann.line_start..middle)
                    .rev()
                    .filter_map(|line| file.get_line(line - 1).map(|s| (line + 1, s)))
                    .find(|(_, s)| filter(s))
                    .map(|(line, _)| line)
                    .unwrap_or(ann.line_start);
                for line in ann.line_start + 1..until {
                    // Every `|` that joins the beginning of the span (`___^`) to the end (`|__^`).
                    add_annotation_to_file(&mut output, Arc::clone(&file), line, ann.as_line());
                }
                let line_end = ann.line_end - 1;
                let end_is_empty = file.get_line(line_end - 1).is_some_and(|s| !filter(&s));
                if middle < line_end && !end_is_empty {
                    add_annotation_to_file(&mut output, Arc::clone(&file), line_end, ann.as_line());
                }
            } else {
                end_ann.annotation_type = AnnotationType::Singleline;
            }
            add_annotation_to_file(&mut output, file, ann.line_end, end_ann);
        }
        for file_vec in output.iter_mut() {
            file_vec.multiline_depth = max_depth;
        }
        output
    }
}

// instead of taking the String length or dividing by 10 while > 0, we multiply a limit by 10 until
// we're higher. If the loop isn't exited by the `return`, the last multiplication will wrap, which
// is OK, because while we cannot fit a higher power of 10 in a usize, the loop will end anyway.
// This is also why we need the max number of decimal digits within a `usize`.
fn num_decimal_digits(num: usize) -> usize {
    #[cfg(target_pointer_width = "64")]
    const MAX_DIGITS: usize = 20;

    #[cfg(target_pointer_width = "32")]
    const MAX_DIGITS: usize = 10;

    #[cfg(target_pointer_width = "16")]
    const MAX_DIGITS: usize = 5;

    let mut lim = 10;
    for num_digits in 1..MAX_DIGITS {
        if num < lim {
            return num_digits;
        }
        lim = lim.wrapping_mul(10);
    }
    MAX_DIGITS
}

// We replace some characters so the CLI output is always consistent and underlines aligned.
// Keep the following list in sync with `rustc_span::char_width`.
const OUTPUT_REPLACEMENTS: &[(char, &str)] = &[
    // In terminals without Unicode support the following will be garbled, but in *all* terminals
    // the underlying codepoint will be as well. We could gate this replacement behind a "unicode
    // support" gate.
    ('\0', "␀"),
    ('\u{0001}', "␁"),
    ('\u{0002}', "␂"),
    ('\u{0003}', "␃"),
    ('\u{0004}', "␄"),
    ('\u{0005}', "␅"),
    ('\u{0006}', "␆"),
    ('\u{0007}', "␇"),
    ('\u{0008}', "␈"),
    ('\t', "    "), // We do our own tab replacement
    ('\u{000b}', "␋"),
    ('\u{000c}', "␌"),
    ('\u{000d}', "␍"),
    ('\u{000e}', "␎"),
    ('\u{000f}', "␏"),
    ('\u{0010}', "␐"),
    ('\u{0011}', "␑"),
    ('\u{0012}', "␒"),
    ('\u{0013}', "␓"),
    ('\u{0014}', "␔"),
    ('\u{0015}', "␕"),
    ('\u{0016}', "␖"),
    ('\u{0017}', "␗"),
    ('\u{0018}', "␘"),
    ('\u{0019}', "␙"),
    ('\u{001a}', "␚"),
    ('\u{001b}', "␛"),
    ('\u{001c}', "␜"),
    ('\u{001d}', "␝"),
    ('\u{001e}', "␞"),
    ('\u{001f}', "␟"),
    ('\u{007f}', "␡"),
    ('\u{200d}', ""), // Replace ZWJ for consistent terminal output of grapheme clusters.
    ('\u{202a}', "�"), // The following unicode text flow control characters are inconsistently
    ('\u{202b}', "�"), // supported across CLIs and can cause confusion due to the bytes on disk
    ('\u{202c}', "�"), // not corresponding to the visible source code, so we replace them always.
    ('\u{202d}', "�"),
    ('\u{202e}', "�"),
    ('\u{2066}', "�"),
    ('\u{2067}', "�"),
    ('\u{2068}', "�"),
    ('\u{2069}', "�"),
];

fn normalize_whitespace(s: &str) -> String {
    const {
        let mut i = 1;
        while i < OUTPUT_REPLACEMENTS.len() {
            assert!(
                OUTPUT_REPLACEMENTS[i - 1].0 < OUTPUT_REPLACEMENTS[i].0,
                "The OUTPUT_REPLACEMENTS array must be sorted (for binary search to work) \
                and must contain no duplicate entries"
            );
            i += 1;
        }
    }
    // Scan the input string for a character in the ordered table above.
    // If it's present, replace it with its alternative string (it can be more than 1 char!).
    // Otherwise, retain the input char.
    s.chars().fold(String::with_capacity(s.len()), |mut s, c| {
        match OUTPUT_REPLACEMENTS.binary_search_by_key(&c, |(k, _)| *k) {
            Ok(i) => s.push_str(OUTPUT_REPLACEMENTS[i].1),
            _ => s.push(c),
        }
        s
    })
}

fn num_overlap(
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
    inclusive: bool,
) -> bool {
    let extra = if inclusive { 1 } else { 0 };
    (b_start..b_end + extra).contains(&a_start) || (a_start..a_end + extra).contains(&b_start)
}

fn overlaps(a1: &Annotation, a2: &Annotation, padding: usize) -> bool {
    num_overlap(
        a1.start_col.display,
        a1.end_col.display + padding,
        a2.start_col.display,
        a2.end_col.display,
        false,
    )
}

fn emit_to_destination(
    rendered_buffer: &[Vec<StyledString>],
    lvl: &Level,
    dst: &mut Destination,
    short_message: bool,
) -> io::Result<()> {
    use crate::lock;

    // In order to prevent error message interleaving, where multiple error lines get intermixed
    // when multiple compiler processes error simultaneously, we emit errors with additional
    // steps.
    //
    // On Unix systems, we write into a buffered terminal rather than directly to a terminal. When
    // the .flush() is called we take the buffer created from the buffered writes and write it at
    // one shot. Because the Unix systems use ANSI for the colors, which is a text-based styling
    // scheme, this buffered approach works and maintains the styling.
    //
    // On Windows, styling happens through calls to a terminal API. This prevents us from using the
    // same buffering approach. Instead, we use a global Windows mutex, which we acquire long
    // enough to output the full error message, then we release.
    let _buffer_lock = lock::acquire_global_lock("rustc_errors");
    for (pos, line) in rendered_buffer.iter().enumerate() {
        for part in line {
            let style = part.style.color_spec(*lvl);
            dst.set_color(&style)?;
            write!(dst, "{}", part.text)?;
            dst.reset()?;
        }
        if !short_message && (!lvl.is_failure_note() || pos != rendered_buffer.len() - 1) {
            writeln!(dst)?;
        }
    }
    dst.flush()?;
    Ok(())
}

pub type Destination = Box<dyn WriteColor + Send>;

struct Buffy {
    buffer_writer: BufferWriter,
    buffer: Buffer,
}

impl Write for Buffy {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buffer_writer.print(&self.buffer)?;
        self.buffer.clear();
        Ok(())
    }
}

impl Drop for Buffy {
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            self.flush().unwrap();
            panic!("buffers need to be flushed in order to print their contents");
        }
    }
}

impl WriteColor for Buffy {
    fn supports_color(&self) -> bool {
        self.buffer.supports_color()
    }

    fn set_color(&mut self, spec: &ColorSpec) -> io::Result<()> {
        self.buffer.set_color(spec)
    }

    fn reset(&mut self) -> io::Result<()> {
        self.buffer.reset()
    }
}

pub fn stderr_destination(color: ColorConfig) -> Destination {
    let choice = color.to_color_choice();
    // On Windows we'll be performing global synchronization on the entire
    // system for emitting rustc errors, so there's no need to buffer
    // anything.
    //
    // On non-Windows we rely on the atomicity of `write` to ensure errors
    // don't get all jumbled up.
    if cfg!(windows) {
        Box::new(StandardStream::stderr(choice))
    } else {
        let buffer_writer = BufferWriter::stderr(choice);
        let buffer = buffer_writer.buffer();
        Box::new(Buffy { buffer_writer, buffer })
    }
}

/// On Windows, BRIGHT_BLUE is hard to read on black. Use cyan instead.
///
/// See #36178.
const BRIGHT_BLUE: Color = if cfg!(windows) { Color::Cyan } else { Color::Blue };

impl Style {
    fn color_spec(&self, lvl: Level) -> ColorSpec {
        let mut spec = ColorSpec::new();
        match self {
            Style::Addition => {
                spec.set_fg(Some(Color::Green)).set_intense(true);
            }
            Style::Removal => {
                spec.set_fg(Some(Color::Red)).set_intense(true);
            }
            Style::LineAndColumn => {}
            Style::LineNumber => {
                spec.set_bold(true);
                spec.set_intense(true);
                spec.set_fg(Some(BRIGHT_BLUE));
            }
            Style::Quotation => {}
            Style::MainHeaderMsg => {
                spec.set_bold(true);
                if cfg!(windows) {
                    spec.set_intense(true).set_fg(Some(Color::White));
                }
            }
            Style::UnderlinePrimary | Style::LabelPrimary => {
                spec = lvl.color();
                spec.set_bold(true);
            }
            Style::UnderlineSecondary | Style::LabelSecondary => {
                spec.set_bold(true).set_intense(true);
                spec.set_fg(Some(BRIGHT_BLUE));
            }
            Style::HeaderMsg | Style::NoStyle => {}
            Style::Level(lvl) => {
                spec = lvl.color();
                spec.set_bold(true);
            }
            Style::Highlight => {
                spec.set_bold(true).set_fg(Some(Color::Magenta));
            }
        }
        spec
    }
}

/// Whether the original and suggested code are the same.
pub fn is_different(sm: &SourceMap, suggested: &str, sp: Span) -> bool {
    let found = match sm.span_to_snippet(sp) {
        Ok(snippet) => snippet,
        Err(e) => {
            warn!(error = ?e, "Invalid span {:?}", sp);
            return true;
        }
    };
    found != suggested
}

/// Whether the original and suggested code are visually similar enough to warrant extra wording.
pub fn detect_confusion_type(sm: &SourceMap, suggested: &str, sp: Span) -> ConfusionType {
    let found = match sm.span_to_snippet(sp) {
        Ok(snippet) => snippet,
        Err(e) => {
            warn!(error = ?e, "Invalid span {:?}", sp);
            return ConfusionType::None;
        }
    };

    let mut has_case_confusion = false;
    let mut has_digit_letter_confusion = false;

    if found.len() == suggested.len() {
        let mut has_case_diff = false;
        let mut has_digit_letter_confusable = false;
        let mut has_other_diff = false;

        let ascii_confusables = &['c', 'f', 'i', 'k', 'o', 's', 'u', 'v', 'w', 'x', 'y', 'z'];

        let digit_letter_confusables = [('0', 'O'), ('1', 'l'), ('5', 'S'), ('8', 'B'), ('9', 'g')];

        for (f, s) in iter::zip(found.chars(), suggested.chars()) {
            if f != s {
                if f.eq_ignore_ascii_case(&s) {
                    // Check for case differences (any character that differs only in case)
                    if ascii_confusables.contains(&f) || ascii_confusables.contains(&s) {
                        has_case_diff = true;
                    } else {
                        has_other_diff = true;
                    }
                } else if digit_letter_confusables.contains(&(f, s))
                    || digit_letter_confusables.contains(&(s, f))
                {
                    // Check for digit-letter confusables (like 0 vs O, 1 vs l, etc.)
                    has_digit_letter_confusable = true;
                } else {
                    has_other_diff = true;
                }
            }
        }

        // If we have case differences and no other differences
        if has_case_diff && !has_other_diff && found != suggested {
            has_case_confusion = true;
        }
        if has_digit_letter_confusable && !has_other_diff && found != suggested {
            has_digit_letter_confusion = true;
        }
    }

    match (has_case_confusion, has_digit_letter_confusion) {
        (true, true) => ConfusionType::Both,
        (true, false) => ConfusionType::Case,
        (false, true) => ConfusionType::DigitLetter,
        (false, false) => ConfusionType::None,
    }
}

/// Represents the type of confusion detected between original and suggested code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfusionType {
    /// No confusion detected
    None,
    /// Only case differences (e.g., "hello" vs "Hello")
    Case,
    /// Only digit-letter confusion (e.g., "0" vs "O", "1" vs "l")
    DigitLetter,
    /// Both case and digit-letter confusion
    Both,
}

impl ConfusionType {
    /// Returns the appropriate label text for this confusion type.
    pub fn label_text(&self) -> &'static str {
        match self {
            ConfusionType::None => "",
            ConfusionType::Case => " (notice the capitalization)",
            ConfusionType::DigitLetter => " (notice the digit/letter confusion)",
            ConfusionType::Both => " (notice the capitalization and digit/letter confusion)",
        }
    }

    /// Combines two confusion types. If either is `Both`, the result is `Both`.
    /// If one is `Case` and the other is `DigitLetter`, the result is `Both`.
    /// Otherwise, returns the non-`None` type, or `None` if both are `None`.
    pub fn combine(self, other: ConfusionType) -> ConfusionType {
        match (self, other) {
            (ConfusionType::None, other) => other,
            (this, ConfusionType::None) => this,
            (ConfusionType::Both, _) | (_, ConfusionType::Both) => ConfusionType::Both,
            (ConfusionType::Case, ConfusionType::DigitLetter)
            | (ConfusionType::DigitLetter, ConfusionType::Case) => ConfusionType::Both,
            (ConfusionType::Case, ConfusionType::Case) => ConfusionType::Case,
            (ConfusionType::DigitLetter, ConfusionType::DigitLetter) => ConfusionType::DigitLetter,
        }
    }

    /// Returns true if this confusion type represents any kind of confusion.
    pub fn has_confusion(&self) -> bool {
        *self != ConfusionType::None
    }
}

pub(crate) fn should_show_source_code(
    ignored_directories: &[String],
    sm: &SourceMap,
    file: &SourceFile,
) -> bool {
    if !sm.ensure_source_file_source_present(file) {
        return false;
    }

    let FileName::Real(name) = &file.name else { return true };
    name.local_path()
        .map(|path| ignored_directories.iter().all(|dir| !path.starts_with(dir)))
        .unwrap_or(true)
}
