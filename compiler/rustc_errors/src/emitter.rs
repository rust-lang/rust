//! The current rustc diagnostics emitter.
//!
//! An `Emitter` takes care of generating the output from a `DiagnosticBuilder` struct.
//!
//! There are various `Emitter` implementations that generate different output formats such as
//! JSON and human readable output.
//!
//! The output types are defined in `rustc_session::config::ErrorOutputType`.

use Destination::*;

use rustc_span::source_map::SourceMap;
use rustc_span::{FileLines, SourceFile, Span};

use crate::snippet::{
    Annotation, AnnotationColumn, AnnotationType, Line, MultilineAnnotation, Style, StyledString,
};
use crate::styled_buffer::StyledBuffer;
use crate::translation::{to_fluent_args, Translate};
use crate::{
    diagnostic::DiagnosticLocation, CodeSuggestion, Diagnostic, DiagnosticId, DiagnosticMessage,
    FluentBundle, Handler, LazyFallbackBundle, Level, MultiSpan, SubDiagnostic,
    SubstitutionHighlight, SuggestionStyle, TerminalUrl,
};
use rustc_lint_defs::pluralize;

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::sync::Lrc;
use rustc_error_messages::{FluentArgs, SpanLabel};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use std::borrow::Cow;
use std::cmp::{max, min, Reverse};
use std::error::Report;
use std::io::prelude::*;
use std::io::{self, IsTerminal};
use std::iter;
use std::path::Path;
use termcolor::{Ansi, BufferWriter, ColorChoice, ColorSpec, StandardStream};
use termcolor::{Buffer, Color, WriteColor};

/// Default column width, used in tests and when terminal dimensions cannot be determined.
const DEFAULT_COLUMN_WIDTH: usize = 140;

/// Describes the way the content of the `rendered` field of the json output is generated
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanReadableErrorType {
    Default(ColorConfig),
    AnnotateSnippet(ColorConfig),
    Short(ColorConfig),
}

impl HumanReadableErrorType {
    /// Returns a (`short`, `color`) tuple
    pub fn unzip(self) -> (bool, ColorConfig) {
        match self {
            HumanReadableErrorType::Default(cc) => (false, cc),
            HumanReadableErrorType::Short(cc) => (true, cc),
            HumanReadableErrorType::AnnotateSnippet(cc) => (false, cc),
        }
    }
    pub fn new_emitter(
        self,
        dst: Box<dyn Write + Send>,
        source_map: Option<Lrc<SourceMap>>,
        bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        teach: bool,
        diagnostic_width: Option<usize>,
        macro_backtrace: bool,
        track_diagnostics: bool,
        terminal_url: TerminalUrl,
    ) -> EmitterWriter {
        let (short, color_config) = self.unzip();
        let color = color_config.suggests_using_colors();
        EmitterWriter::new(
            dst,
            source_map,
            bundle,
            fallback_bundle,
            short,
            teach,
            color,
            diagnostic_width,
            macro_backtrace,
            track_diagnostics,
            terminal_url,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Margin {
    /// The available whitespace in the left that can be consumed when centering.
    pub whitespace_left: usize,
    /// The column of the beginning of left-most span.
    pub span_left: usize,
    /// The column of the end of right-most span.
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

    fn was_cut_right(&self, line_len: usize) -> bool {
        let right =
            if self.computed_right == self.span_right || self.computed_right == self.label_right {
                // Account for the "..." padding given above. Otherwise we end up with code lines that
                // do fit but end in "..." as if they were trimmed.
                self.computed_right - 6
            } else {
                self.computed_right
            };
        right < line_len && self.computed_left + self.column_width < line_len
    }

    fn compute(&mut self, max_line_len: usize) {
        // When there's a lot of whitespace (>20), we want to trim it as it is useless.
        self.computed_left = if self.whitespace_left > 20 {
            self.whitespace_left - 16 // We want some padding.
        } else {
            0
        };
        // We want to show as much as possible, max_line_len is the right-most boundary for the
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

const ANONYMIZED_LINE_NUM: &str = "LL";

/// Emitter trait for emitting errors.
pub trait Emitter: Translate {
    /// Emit a structured diagnostic.
    fn emit_diagnostic(&mut self, diag: &Diagnostic);

    /// Emit a notification that an artifact has been output.
    /// This is currently only supported for the JSON format,
    /// other formats can, and will, simply ignore it.
    fn emit_artifact_notification(&mut self, _path: &Path, _artifact_type: &str) {}

    fn emit_future_breakage_report(&mut self, _diags: Vec<Diagnostic>) {}

    /// Emit list of unused externs
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

    fn source_map(&self) -> Option<&Lrc<SourceMap>>;

    /// Formats the substitutions of the primary_span
    ///
    /// There are a lot of conditions to this method, but in short:
    ///
    /// * If the current `Diagnostic` has only one visible `CodeSuggestion`,
    ///   we format the `help` suggestion depending on the content of the
    ///   substitutions. In that case, we return the modified span only.
    ///
    /// * If the current `Diagnostic` has multiple suggestions,
    ///   we return the original `primary_span` and the original suggestions.
    fn primary_span_formatted<'a>(
        &mut self,
        diag: &'a Diagnostic,
        fluent_args: &FluentArgs<'_>,
    ) -> (MultiSpan, &'a [CodeSuggestion]) {
        let mut primary_span = diag.span.clone();
        let suggestions = diag.suggestions.as_deref().unwrap_or(&[]);
        if let Some((sugg, rest)) = suggestions.split_first() {
            let msg = self.translate_message(&sugg.msg, fluent_args).map_err(Report::new).unwrap();
            if rest.is_empty() &&
               // ^ if there is only one suggestion
               // don't display multi-suggestions as labels
               sugg.substitutions.len() == 1 &&
               // don't display multipart suggestions as labels
               sugg.substitutions[0].parts.len() == 1 &&
               // don't display long messages as labels
               msg.split_whitespace().count() < 10 &&
               // don't display multiline suggestions as labels
               !sugg.substitutions[0].parts[0].snippet.contains('\n') &&
               ![
                    // when this style is set we want the suggestion to be a message, not inline
                    SuggestionStyle::HideCodeAlways,
                    // trivial suggestion for tooling's sake, never shown
                    SuggestionStyle::CompletelyHidden,
                    // subtle suggestion, never shown inline
                    SuggestionStyle::ShowAlways,
               ].contains(&sugg.style)
            {
                let substitution = &sugg.substitutions[0].parts[0].snippet.trim();
                let msg = if substitution.is_empty() || sugg.style.hide_inline() {
                    // This substitution is only removal OR we explicitly don't want to show the
                    // code inline (`hide_inline`). Therefore, we don't show the substitution.
                    format!("help: {}", &msg)
                } else {
                    // Show the default suggestion text with the substitution
                    format!(
                        "help: {}{}: `{}`",
                        &msg,
                        if self.source_map().is_some_and(|sm| is_case_difference(
                            sm,
                            substitution,
                            sugg.substitutions[0].parts[0].span,
                        )) {
                            " (notice the capitalization)"
                        } else {
                            ""
                        },
                        substitution,
                    )
                };
                primary_span.push_span_label(sugg.substitutions[0].parts[0].span, msg);

                // We return only the modified primary_span
                (primary_span, &[])
            } else {
                // if there are multiple suggestions, print them all in full
                // to be consistent. We could try to figure out if we can
                // make one (or the first one) inline, but that would give
                // undue importance to a semi-random suggestion
                (primary_span, suggestions)
            }
        } else {
            (primary_span, suggestions)
        }
    }

    fn fix_multispans_in_extern_macros_and_render_macro_backtrace(
        &self,
        span: &mut MultiSpan,
        children: &mut Vec<SubDiagnostic>,
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

                    ExpnKind::Macro(macro_kind, name) => Some((macro_kind, name)),
                }
            })
            .collect();

        if !backtrace {
            self.fix_multispans_in_extern_macros(span, children);
        }

        self.render_multispans_macro_backtrace(span, children, backtrace);

        if !backtrace {
            if let Some((macro_kind, name)) = has_macro_spans.first() {
                // Mark the actual macro this originates from
                let and_then = if let Some((macro_kind, last_name)) = has_macro_spans.last()
                    && last_name != name
                {
                    let descr = macro_kind.descr();
                    format!(
                        " which comes from the expansion of the {descr} `{last_name}`",
                    )
                } else {
                    "".to_string()
                };

                let descr = macro_kind.descr();
                let msg = format!(
                    "this {level} originates in the {descr} `{name}`{and_then} \
                    (in Nightly builds, run with -Z macro-backtrace for more info)",
                );

                children.push(SubDiagnostic {
                    level: Level::Note,
                    message: vec![(DiagnosticMessage::from(msg), Style::NoStyle)],
                    span: MultiSpan::new(),
                    render_span: None,
                });
            }
        }
    }

    fn render_multispans_macro_backtrace(
        &self,
        span: &mut MultiSpan,
        children: &mut Vec<SubDiagnostic>,
        backtrace: bool,
    ) {
        for span in iter::once(span).chain(children.iter_mut().map(|child| &mut child.span)) {
            self.render_multispan_macro_backtrace(span, backtrace);
        }
    }

    fn render_multispan_macro_backtrace(&self, span: &mut MultiSpan, always_backtrace: bool) {
        let mut new_labels: Vec<(Span, String)> = vec![];

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
                    new_labels.push((
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
                            "this procedural macro expansion".into()
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
                    new_labels.push((
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
    fn fix_multispans_in_extern_macros(
        &self,
        span: &mut MultiSpan,
        children: &mut Vec<SubDiagnostic>,
    ) {
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

impl Translate for EmitterWriter {
    fn fluent_bundle(&self) -> Option<&Lrc<FluentBundle>> {
        self.fluent_bundle.as_ref()
    }

    fn fallback_fluent_bundle(&self) -> &FluentBundle {
        &self.fallback_bundle
    }
}

impl Emitter for EmitterWriter {
    fn source_map(&self) -> Option<&Lrc<SourceMap>> {
        self.sm.as_ref()
    }

    fn emit_diagnostic(&mut self, diag: &Diagnostic) {
        let fluent_args = to_fluent_args(diag.args());

        let mut children = diag.children.clone();
        let (mut primary_span, suggestions) = self.primary_span_formatted(diag, &fluent_args);
        debug!("emit_diagnostic: suggestions={:?}", suggestions);

        self.fix_multispans_in_extern_macros_and_render_macro_backtrace(
            &mut primary_span,
            &mut children,
            &diag.level,
            self.macro_backtrace,
        );

        self.emit_messages_default(
            &diag.level,
            &diag.message,
            &fluent_args,
            &diag.code,
            &primary_span,
            &children,
            suggestions,
            self.track_diagnostics.then_some(&diag.emitted_at),
        );
    }

    fn should_show_explain(&self) -> bool {
        !self.short_message
    }

    fn supports_color(&self) -> bool {
        self.dst.supports_color()
    }
}

/// An emitter that does nothing when emitting a non-fatal diagnostic.
/// Fatal diagnostics are forwarded to `fatal_handler` to avoid silent
/// failures of rustc, as witnessed e.g. in issue #89358.
pub struct SilentEmitter {
    pub fatal_handler: Handler,
    pub fatal_note: Option<String>,
}

impl Translate for SilentEmitter {
    fn fluent_bundle(&self) -> Option<&Lrc<FluentBundle>> {
        None
    }

    fn fallback_fluent_bundle(&self) -> &FluentBundle {
        panic!("silent emitter attempted to translate message")
    }
}

impl Emitter for SilentEmitter {
    fn source_map(&self) -> Option<&Lrc<SourceMap>> {
        None
    }

    fn emit_diagnostic(&mut self, d: &Diagnostic) {
        if d.level == Level::Fatal {
            let mut d = d.clone();
            if let Some(ref note) = self.fatal_note {
                d.note(note.clone());
            }
            self.fatal_handler.emit_diagnostic(&mut d);
        }
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
    fn to_color_choice(self) -> ColorChoice {
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
    fn suggests_using_colors(self) -> bool {
        match self {
            ColorConfig::Always | ColorConfig::Auto => true,
            ColorConfig::Never => false,
        }
    }
}

/// Handles the writing of `HumanReadableErrorType::Default` and `HumanReadableErrorType::Short`
pub struct EmitterWriter {
    dst: Destination,
    sm: Option<Lrc<SourceMap>>,
    fluent_bundle: Option<Lrc<FluentBundle>>,
    fallback_bundle: LazyFallbackBundle,
    short_message: bool,
    teach: bool,
    ui_testing: bool,
    diagnostic_width: Option<usize>,

    macro_backtrace: bool,
    track_diagnostics: bool,
    terminal_url: TerminalUrl,
}

#[derive(Debug)]
pub struct FileWithAnnotatedLines {
    pub file: Lrc<SourceFile>,
    pub lines: Vec<Line>,
    multiline_depth: usize,
}

impl EmitterWriter {
    pub fn stderr(
        color_config: ColorConfig,
        source_map: Option<Lrc<SourceMap>>,
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        short_message: bool,
        teach: bool,
        diagnostic_width: Option<usize>,
        macro_backtrace: bool,
        track_diagnostics: bool,
        terminal_url: TerminalUrl,
    ) -> EmitterWriter {
        let dst = Destination::from_stderr(color_config);
        EmitterWriter {
            dst,
            sm: source_map,
            fluent_bundle,
            fallback_bundle,
            short_message,
            teach,
            ui_testing: false,
            diagnostic_width,
            macro_backtrace,
            track_diagnostics,
            terminal_url,
        }
    }

    pub fn new(
        dst: Box<dyn Write + Send>,
        source_map: Option<Lrc<SourceMap>>,
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        short_message: bool,
        teach: bool,
        colored: bool,
        diagnostic_width: Option<usize>,
        macro_backtrace: bool,
        track_diagnostics: bool,
        terminal_url: TerminalUrl,
    ) -> EmitterWriter {
        EmitterWriter {
            dst: Raw(dst, colored),
            sm: source_map,
            fluent_bundle,
            fallback_bundle,
            short_message,
            teach,
            ui_testing: false,
            diagnostic_width,
            macro_backtrace,
            track_diagnostics,
            terminal_url,
        }
    }

    pub fn ui_testing(mut self, ui_testing: bool) -> Self {
        self.ui_testing = ui_testing;
        self
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
    ) {
        // Tabs are assumed to have been replaced by spaces in calling code.
        debug_assert!(!source_string.contains('\t'));
        let line_len = source_string.len();
        // Create the source line we will highlight.
        let left = margin.left(line_len);
        let right = margin.right(line_len);
        // On long lines, we strip the source line, accounting for unicode.
        let mut taken = 0;
        let code: String = source_string
            .chars()
            .skip(left)
            .take_while(|ch| {
                // Make sure that the trimming on the right will fall within the terminal width.
                // FIXME: `unicode_width` sometimes disagrees with terminals on how wide a `char` is.
                // For now, just accept that sometimes the code line will be longer than desired.
                let next = unicode_width::UnicodeWidthChar::width(*ch).unwrap_or(1);
                if taken + next > right - left {
                    return false;
                }
                taken += next;
                true
            })
            .collect();
        buffer.puts(line_offset, code_offset, &code, Style::Quotation);
        if margin.was_cut_left() {
            // We have stripped some code/whitespace from the beginning, make it clear.
            buffer.puts(line_offset, code_offset, "...", Style::LineNumber);
        }
        if margin.was_cut_right(line_len) {
            // We have stripped some code after the right-most span end, make it clear we did so.
            buffer.puts(line_offset, code_offset + taken - 3, "...", Style::LineNumber);
        }
        buffer.puts(line_offset, 0, &self.maybe_anonymized(line_index), Style::LineNumber);

        draw_col_separator_no_space(buffer, line_offset, width_offset - 2);
    }

    #[instrument(level = "trace", skip(self), ret)]
    fn render_source_line(
        &self,
        buffer: &mut StyledBuffer,
        file: Lrc<SourceFile>,
        line: &Line,
        width_offset: usize,
        code_offset: usize,
        margin: Margin,
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

        let source_string = match file.get_line(line.line_index - 1) {
            Some(s) => normalize_whitespace(&s),
            None => return Vec::new(),
        };
        trace!(?source_string);

        let line_offset = buffer.num_lines();

        // Left trim
        let left = margin.left(source_string.len());

        // Account for unicode characters of width !=0 that were removed.
        let left = source_string
            .chars()
            .take(left)
            .map(|ch| unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1))
            .sum();

        self.draw_line(
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
                if source_string.chars().take(ann.start_col.display).all(|c| c.is_whitespace()) {
                    let style = if ann.is_primary {
                        Style::UnderlinePrimary
                    } else {
                        Style::UnderlineSecondary
                    };
                    annotations.push((depth, style));
                    buffer_ops.push((line_offset, width_offset + depth - 1, '/', style));
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
        let mut annotations_position = vec![];
        let mut line_len = 0;
        let mut p = 0;
        for (i, annotation) in annotations.iter().enumerate() {
            for (j, next) in annotations.iter().enumerate() {
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
            draw_col_separator(buffer, line_offset + pos + 1, width_offset - 2);
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
            let style = if annotation.is_primary {
                Style::UnderlinePrimary
            } else {
                Style::UnderlineSecondary
            };
            let pos = pos + 1;
            match annotation.annotation_type {
                AnnotationType::MultilineStart(depth) | AnnotationType::MultilineEnd(depth) => {
                    draw_range(
                        buffer,
                        '_',
                        line_offset + pos,
                        width_offset + depth,
                        (code_offset + annotation.start_col.display).saturating_sub(left),
                        style,
                    );
                }
                _ if self.teach => {
                    buffer.set_style_range(
                        line_offset,
                        (code_offset + annotation.start_col.display).saturating_sub(left),
                        (code_offset + annotation.end_col.display).saturating_sub(left),
                        style,
                        annotation.is_primary,
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
            let style = if annotation.is_primary {
                Style::UnderlinePrimary
            } else {
                Style::UnderlineSecondary
            };
            let pos = pos + 1;

            if pos > 1 && (annotation.has_label() || annotation.takes_space()) {
                for p in line_offset + 1..=line_offset + pos {
                    buffer.putc(
                        p,
                        (code_offset + annotation.start_col.display).saturating_sub(left),
                        '|',
                        style,
                    );
                }
            }
            match annotation.annotation_type {
                AnnotationType::MultilineStart(depth) => {
                    for p in line_offset + pos + 1..line_offset + line_len + 2 {
                        buffer.putc(p, width_offset + depth - 1, '|', style);
                    }
                }
                AnnotationType::MultilineEnd(depth) => {
                    for p in line_offset..=line_offset + pos {
                        buffer.putc(p, width_offset + depth - 1, '|', style);
                    }
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
                (pos + 1, (annotation.end_col.display + 1).saturating_sub(left))
            } else {
                (pos + 2, annotation.start_col.display.saturating_sub(left))
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
        for &(_, annotation) in &annotations_position {
            let (underline, style) = if annotation.is_primary {
                ('^', Style::UnderlinePrimary)
            } else {
                ('-', Style::UnderlineSecondary)
            };
            for p in annotation.start_col.display..annotation.end_col.display {
                buffer.putc(
                    line_offset + 1,
                    (code_offset + p).saturating_sub(left),
                    underline,
                    style,
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
                sm.ensure_source_file_source_present(file)
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

    fn get_max_line_num(&mut self, span: &MultiSpan, children: &[SubDiagnostic]) -> usize {
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
    fn msg_to_buffer(
        &self,
        buffer: &mut StyledBuffer,
        msg: &[(DiagnosticMessage, Style)],
        args: &FluentArgs<'_>,
        padding: usize,
        label: &str,
        override_style: Option<Style>,
    ) {
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
        //     let msg = vec![
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
        for (text, style) in msg.iter() {
            let text = self.translate_message(text, args).map_err(Report::new).unwrap();
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
                buffer.append(line_number, &text, style_or_override(*style, override_style));
            }
        }
    }

    #[instrument(level = "trace", skip(self, args), ret)]
    fn emit_message_default(
        &mut self,
        msp: &MultiSpan,
        msg: &[(DiagnosticMessage, Style)],
        args: &FluentArgs<'_>,
        code: &Option<DiagnosticId>,
        level: &Level,
        max_line_num_len: usize,
        is_secondary: bool,
        emitted_at: Option<&DiagnosticLocation>,
    ) -> io::Result<()> {
        let mut buffer = StyledBuffer::new();

        if !msp.has_primary_spans() && !msp.has_span_labels() && is_secondary && !self.short_message
        {
            // This is a secondary message with no span info
            for _ in 0..max_line_num_len {
                buffer.prepend(0, " ", Style::NoStyle);
            }
            draw_note_separator(&mut buffer, 0, max_line_num_len + 1);
            if *level != Level::FailureNote {
                buffer.append(0, level.to_str(), Style::MainHeaderMsg);
                buffer.append(0, ": ", Style::NoStyle);
            }
            self.msg_to_buffer(&mut buffer, msg, args, max_line_num_len, "note", None);
        } else {
            let mut label_width = 0;
            // The failure note level itself does not provide any useful diagnostic information
            if *level != Level::FailureNote {
                buffer.append(0, level.to_str(), Style::Level(*level));
                label_width += level.to_str().len();
            }
            // only render error codes, not lint codes
            if let Some(DiagnosticId::Error(ref code)) = *code {
                buffer.append(0, "[", Style::Level(*level));
                let code = if let TerminalUrl::Yes = self.terminal_url {
                    let path = "https://doc.rust-lang.org/error_codes";
                    format!("\x1b]8;;{path}/{code}.html\x07{code}\x1b]8;;\x07")
                } else {
                    code.clone()
                };
                buffer.append(0, &code, Style::Level(*level));
                buffer.append(0, "]", Style::Level(*level));
                label_width += 2 + code.len();
            }
            let header_style = if is_secondary { Style::HeaderMsg } else { Style::MainHeaderMsg };
            if *level != Level::FailureNote {
                buffer.append(0, ": ", header_style);
                label_width += 2;
            }
            for (text, _) in msg.iter() {
                let text = self.translate_message(text, args).map_err(Report::new).unwrap();
                // Account for newlines to align output to its label.
                for (line, text) in normalize_whitespace(&text).lines().enumerate() {
                    buffer.append(
                        line,
                        &format!(
                            "{}{}",
                            if line == 0 { String::new() } else { " ".repeat(label_width) },
                            text
                        ),
                        header_style,
                    );
                }
            }
        }
        let mut annotated_files = FileWithAnnotatedLines::collect_annotations(self, args, msp);
        trace!("{annotated_files:#?}");

        // Make sure our primary file comes first
        let primary_span = msp.primary_span().unwrap_or_default();
        let (Some(sm), false) = (self.sm.as_ref(), primary_span.is_dummy()) else {
            // If we don't have span information, emit and exit
            return emit_to_destination(&buffer.render(), level, &mut self.dst, self.short_message);
        };
        let primary_lo = sm.lookup_char_pos(primary_span.lo());
        if let Ok(pos) =
            annotated_files.binary_search_by(|x| x.file.name.cmp(&primary_lo.file.name))
        {
            annotated_files.swap(0, pos);
        }

        // Print out the annotate source lines that correspond with the error
        for annotated_file in annotated_files {
            // we can't annotate anything if the source is unavailable.
            if !sm.ensure_source_file_source_present(annotated_file.file.clone()) {
                if !self.short_message {
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
                                buffer.prepend(line_idx, "--> ", Style::LineNumber);
                            } else {
                                buffer.prepend(line_idx, "::: ", Style::LineNumber);
                            }
                            for _ in 0..max_line_num_len {
                                buffer.prepend(line_idx, " ", Style::NoStyle);
                            }
                            line_idx += 1;
                        }
                        for (label, is_primary) in labels.into_iter() {
                            let style = if is_primary {
                                Style::LabelPrimary
                            } else {
                                Style::LabelSecondary
                            };
                            buffer.prepend(line_idx, " |", Style::LineNumber);
                            for _ in 0..max_line_num_len {
                                buffer.prepend(line_idx, " ", Style::NoStyle);
                            }
                            line_idx += 1;
                            buffer.append(line_idx, " = note: ", style);
                            for _ in 0..max_line_num_len {
                                buffer.prepend(line_idx, " ", Style::NoStyle);
                            }
                            buffer.append(line_idx, label, style);
                            line_idx += 1;
                        }
                    }
                }
                continue;
            }

            // print out the span location and spacer before we print the annotated source
            // to do this, we need to know if this span will be primary
            let is_primary = primary_lo.file.name == annotated_file.file.name;
            if is_primary {
                let loc = primary_lo.clone();
                if !self.short_message {
                    // remember where we are in the output buffer for easy reference
                    let buffer_msg_line_offset = buffer.num_lines();

                    buffer.prepend(buffer_msg_line_offset, "--> ", Style::LineNumber);
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

                // Add spacing line
                draw_col_separator_no_space(
                    &mut buffer,
                    buffer_msg_line_offset,
                    max_line_num_len + 1,
                );

                // Then, the secondary file indicator
                buffer.prepend(buffer_msg_line_offset + 1, "::: ", Style::LineNumber);
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
                draw_col_separator_no_space(
                    &mut buffer,
                    buffer_msg_line_offset,
                    max_line_num_len + 1,
                );

                // Contains the vertical lines' positions for active multiline annotations
                let mut multilines = FxIndexMap::default();

                // Get the left-side margin to remove it
                let mut whitespace_margin = usize::MAX;
                for line_idx in 0..annotated_file.lines.len() {
                    let file = annotated_file.file.clone();
                    let line = &annotated_file.lines[line_idx];
                    if let Some(source_string) = file.get_line(line.line_index - 1) {
                        let leading_whitespace = source_string
                            .chars()
                            .take_while(|c| c.is_whitespace())
                            .map(|c| {
                                match c {
                                    // Tabs are displayed as 4 spaces
                                    '\t' => 4,
                                    _ => 1,
                                }
                            })
                            .sum();
                        if source_string.chars().any(|c| !c.is_whitespace()) {
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
                        span_left_margin = min(span_left_margin, ann.start_col.display);
                        span_left_margin = min(span_left_margin, ann.end_col.display);
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
                        annotated_file.file.get_line(line.line_index - 1).map_or(0, |s| s.len()),
                    );
                    for ann in &line.annotations {
                        span_right_margin = max(span_right_margin, ann.start_col.display);
                        span_right_margin = max(span_right_margin, ann.end_col.display);
                        // FIXME: account for labels not in the same line
                        let label_right = ann.label.as_ref().map_or(0, |l| l.len() + 1);
                        label_right_margin =
                            max(label_right_margin, ann.end_col.display + label_right);
                    }
                }

                let width_offset = 3 + max_line_num_len;
                let code_offset = if annotated_file.multiline_depth == 0 {
                    width_offset
                } else {
                    width_offset + annotated_file.multiline_depth + 1
                };

                let column_width = if let Some(width) = self.diagnostic_width {
                    width.saturating_sub(code_offset)
                } else if self.ui_testing {
                    DEFAULT_COLUMN_WIDTH
                } else {
                    termize::dimensions()
                        .map(|(w, _)| w.saturating_sub(code_offset))
                        .unwrap_or(DEFAULT_COLUMN_WIDTH)
                };

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
                        annotated_file.file.clone(),
                        &annotated_file.lines[line_idx],
                        width_offset,
                        code_offset,
                        margin,
                    );

                    let mut to_add = FxHashMap::default();

                    for (depth, style) in depths {
                        if multilines.remove(&depth).is_none() {
                            to_add.insert(depth, style);
                        }
                    }

                    // Set the multiline annotation vertical lines to the left of
                    // the code in this line.
                    for (depth, style) in &multilines {
                        for line in previous_buffer_line..buffer.num_lines() {
                            draw_multiline_line(&mut buffer, line, width_offset, *depth, *style);
                        }
                    }
                    // check to see if we need to print out or elide lines that come between
                    // this annotated line and the next one.
                    if line_idx < (annotated_file.lines.len() - 1) {
                        let line_idx_delta = annotated_file.lines[line_idx + 1].line_index
                            - annotated_file.lines[line_idx].line_index;
                        if line_idx_delta > 2 {
                            let last_buffer_line_num = buffer.num_lines();
                            buffer.puts(last_buffer_line_num, 0, "...", Style::LineNumber);

                            // Set the multiline annotation vertical lines on `...` bridging line.
                            for (depth, style) in &multilines {
                                draw_multiline_line(
                                    &mut buffer,
                                    last_buffer_line_num,
                                    width_offset,
                                    *depth,
                                    *style,
                                );
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
                                draw_multiline_line(
                                    &mut buffer,
                                    last_buffer_line_num,
                                    width_offset,
                                    *depth,
                                    *style,
                                );
                            }
                        }
                    }

                    multilines.extend(&to_add);
                }
            }
            trace!("buffer: {:#?}", buffer.render());
        }

        if let Some(tracked) = emitted_at {
            let track = format!("-Ztrack-diagnostics: created at {tracked}");
            let len = buffer.num_lines();
            buffer.append(len, &track, Style::NoStyle);
        }

        // final step: take our styled buffer, render it, then output it
        emit_to_destination(&buffer.render(), level, &mut self.dst, self.short_message)?;

        Ok(())
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
            // Suggestions coming from macros can have malformed spans. This is a heavy handed
            // approach to avoid ICEs by ignoring the suggestion outright.
            return Ok(());
        }

        let mut buffer = StyledBuffer::new();

        // Render the suggestion message
        buffer.append(0, level.to_str(), Style::Level(*level));
        buffer.append(0, ": ", Style::HeaderMsg);

        self.msg_to_buffer(
            &mut buffer,
            &[(suggestion.msg.to_owned(), Style::NoStyle)],
            args,
            max_line_num_len,
            "suggestion",
            Some(Style::HeaderMsg),
        );

        let mut row_num = 2;
        draw_col_separator_no_space(&mut buffer, 1, max_line_num_len + 1);
        let mut notice_capitalization = false;
        for (complete, parts, highlights, only_capitalization) in
            suggestions.iter().take(MAX_SUGGESTIONS)
        {
            debug!(?complete, ?parts, ?highlights);
            notice_capitalization |= only_capitalization;

            let has_deletion = parts.iter().any(|p| p.is_deletion(sm));
            let is_multiline = complete.lines().count() > 1;

            if let Some(span) = span.primary_span() {
                // Compare the primary span of the diagnostic with the span of the suggestion
                // being emitted. If they belong to the same file, we don't *need* to show the
                // file name, saving in verbosity, but if it *isn't* we do need it, otherwise we're
                // telling users to make a change but not clarifying *where*.
                let loc = sm.lookup_char_pos(parts[0].span.lo());
                if loc.file.name != sm.span_to_filename(span) && loc.file.name.is_real() {
                    let arrow = "--> ";
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
            draw_col_separator_no_space(&mut buffer, row_num - 1, max_line_num_len + 1);
            let mut lines = complete.lines();
            if lines.clone().next().is_none() {
                // Account for a suggestion to completely remove a line(s) with whitespace (#94192).
                let line_end = sm.lookup_char_pos(parts[0].span.hi()).line;
                for line in line_start..=line_end {
                    buffer.puts(
                        row_num - 1 + line - line_start,
                        0,
                        &self.maybe_anonymized(line),
                        Style::LineNumber,
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
                    //   ...
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

                        buffer.puts(row_num, max_line_num_len - 1, "...", Style::LineNumber);
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
            if let DisplaySuggestion::Add = show_code_change && is_item_attribute {
                // The suggestion adds an entire line of code, ending on a newline, so we'll also
                // print the *following* line, to provide context of what we're advising people to
                // do. Otherwise you would only see contextless code that can be confused for
                // already existing code, despite the colors and UI elements.
                // We special case `#[derive(_)]\n` and other attribute suggestions, because those
                // are the ones where context is most useful.
                let file_lines = sm
                    .span_to_lines(span.primary_span().unwrap().shrink_to_hi())
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
                draw_col_separator_no_space(&mut buffer, row_num, max_line_num_len + 1);
                for part in parts {
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
                    let sub_len: usize =
                        if is_whitespace_addition { &part.snippet } else { part.snippet.trim() }
                            .chars()
                            .map(|ch| unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1))
                            .sum();

                    let offset: isize = offsets
                        .iter()
                        .filter_map(
                            |(start, v)| if span_start_pos <= *start { None } else { Some(v) },
                        )
                        .sum();
                    let underline_start = (span_start_pos + start) as isize + offset;
                    let underline_end = (span_start_pos + start + sub_len) as isize + offset;
                    assert!(underline_start >= 0 && underline_end >= 0);
                    let padding: usize = max_line_num_len + 3;
                    for p in underline_start..underline_end {
                        if let DisplaySuggestion::Underline = show_code_change {
                            // If this is a replacement, underline with `^`, if this is an addition
                            // underline with `+`.
                            buffer.putc(
                                row_num,
                                (padding as isize + p) as usize,
                                if part.is_addition(sm) { '+' } else { '~' },
                                Style::Addition,
                            );
                        }
                    }
                    if let DisplaySuggestion::Diff = show_code_change {
                        // Colorize removal with red in diff format.
                        buffer.set_style_range(
                            row_num - 2,
                            (padding as isize + span_start_pos as isize) as usize,
                            (padding as isize + span_end_pos as isize) as usize,
                            Style::Removal,
                            true,
                        );
                    }

                    // length of the code after substitution
                    let full_sub_len = part
                        .snippet
                        .chars()
                        .map(|ch| unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1))
                        .sum::<usize>() as isize;

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
                buffer.puts(row_num, max_line_num_len - 1, "...", Style::LineNumber);
            } else if let DisplaySuggestion::None = show_code_change {
                draw_col_separator_no_space(&mut buffer, row_num, max_line_num_len + 1);
                row_num += 1;
            }
        }
        if suggestions.len() > MAX_SUGGESTIONS {
            let others = suggestions.len() - MAX_SUGGESTIONS;
            let msg = format!("and {} other candidate{}", others, pluralize!(others));
            buffer.puts(row_num, max_line_num_len + 3, &msg, Style::NoStyle);
        } else if notice_capitalization {
            let msg = "notice the capitalization difference";
            buffer.puts(row_num, max_line_num_len + 3, msg, Style::NoStyle);
        }
        emit_to_destination(&buffer.render(), level, &mut self.dst, self.short_message)?;
        Ok(())
    }

    #[instrument(level = "trace", skip(self, args, code, children, suggestions))]
    fn emit_messages_default(
        &mut self,
        level: &Level,
        message: &[(DiagnosticMessage, Style)],
        args: &FluentArgs<'_>,
        code: &Option<DiagnosticId>,
        span: &MultiSpan,
        children: &[SubDiagnostic],
        suggestions: &[CodeSuggestion],
        emitted_at: Option<&DiagnosticLocation>,
    ) {
        let max_line_num_len = if self.ui_testing {
            ANONYMIZED_LINE_NUM.len()
        } else {
            let n = self.get_max_line_num(span, children);
            num_decimal_digits(n)
        };

        match self.emit_message_default(
            span,
            message,
            args,
            code,
            level,
            max_line_num_len,
            false,
            emitted_at,
        ) {
            Ok(()) => {
                if !children.is_empty()
                    || suggestions.iter().any(|s| s.style != SuggestionStyle::CompletelyHidden)
                {
                    let mut buffer = StyledBuffer::new();
                    if !self.short_message {
                        draw_col_separator_no_space(&mut buffer, 0, max_line_num_len + 1);
                    }
                    if let Err(e) = emit_to_destination(
                        &buffer.render(),
                        level,
                        &mut self.dst,
                        self.short_message,
                    ) {
                        panic!("failed to emit error: {}", e)
                    }
                }
                if !self.short_message {
                    for child in children {
                        let span = child.render_span.as_ref().unwrap_or(&child.span);
                        if let Err(err) = self.emit_message_default(
                            span,
                            &child.message,
                            args,
                            &None,
                            &child.level,
                            max_line_num_len,
                            true,
                            None,
                        ) {
                            panic!("failed to emit error: {}", err);
                        }
                    }
                    for sugg in suggestions {
                        match sugg.style {
                            SuggestionStyle::CompletelyHidden => {
                                // do not display this suggestion, it is meant only for tools
                            }
                            SuggestionStyle::HideCodeAlways => {
                                if let Err(e) = self.emit_message_default(
                                    &MultiSpan::new(),
                                    &[(sugg.msg.to_owned(), Style::HeaderMsg)],
                                    args,
                                    &None,
                                    &Level::Help,
                                    max_line_num_len,
                                    true,
                                    None,
                                ) {
                                    panic!("failed to emit error: {}", e);
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
                                    panic!("failed to emit error: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => panic!("failed to emit error: {}", e),
        }

        let mut dst = self.dst.writable();
        match writeln!(dst) {
            Err(e) => panic!("failed to emit error: {}", e),
            _ => {
                if let Err(e) = dst.flush() {
                    panic!("failed to emit error: {}", e)
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
                buffer.puts(
                    *row_num - 1,
                    0,
                    &self.maybe_anonymized(line_num + index),
                    Style::LineNumber,
                );
                buffer.puts(*row_num - 1, max_line_num_len + 1, "- ", Style::Removal);
                let line = normalize_whitespace(
                    &file_lines.file.get_line(line_to_remove.line_index).unwrap(),
                );
                buffer.puts(*row_num - 1, max_line_num_len + 3, &line, Style::NoStyle);
                *row_num += 1;
            }
            // If the last line is exactly equal to the line we need to add, we can skip both of them.
            // This allows us to avoid output like the following:
            // 2 - &
            // 2 + if true { true } else { false }
            // 3 - if true { true } else { false }
            // If those lines aren't equal, we print their diff
            let last_line_index = file_lines.lines[file_lines.lines.len() - 1].line_index;
            let last_line = &file_lines.file.get_line(last_line_index).unwrap();
            if last_line != line_to_add {
                buffer.puts(
                    *row_num - 1,
                    0,
                    &self.maybe_anonymized(line_num + file_lines.lines.len() - 1),
                    Style::LineNumber,
                );
                buffer.puts(*row_num - 1, max_line_num_len + 1, "- ", Style::Removal);
                buffer.puts(
                    *row_num - 1,
                    max_line_num_len + 3,
                    &normalize_whitespace(last_line),
                    Style::NoStyle,
                );
                buffer.puts(*row_num, 0, &self.maybe_anonymized(line_num), Style::LineNumber);
                buffer.puts(*row_num, max_line_num_len + 1, "+ ", Style::Addition);
                buffer.append(*row_num, &normalize_whitespace(line_to_add), Style::NoStyle);
            } else {
                *row_num -= 2;
            }
        } else if is_multiline {
            buffer.puts(*row_num, 0, &self.maybe_anonymized(line_num), Style::LineNumber);
            match &highlight_parts {
                [SubstitutionHighlight { start: 0, end }] if *end == line_to_add.len() => {
                    buffer.puts(*row_num, max_line_num_len + 1, "+ ", Style::Addition);
                }
                [] => {
                    draw_col_separator(buffer, *row_num, max_line_num_len + 1);
                }
                _ => {
                    buffer.puts(*row_num, max_line_num_len + 1, "~ ", Style::Addition);
                }
            }
            buffer.append(*row_num, &normalize_whitespace(line_to_add), Style::NoStyle);
        } else if let DisplaySuggestion::Add = show_code_change {
            buffer.puts(*row_num, 0, &self.maybe_anonymized(line_num), Style::LineNumber);
            buffer.puts(*row_num, max_line_num_len + 1, "+ ", Style::Addition);
            buffer.append(*row_num, &normalize_whitespace(line_to_add), Style::NoStyle);
        } else {
            buffer.puts(*row_num, 0, &self.maybe_anonymized(line_num), Style::LineNumber);
            draw_col_separator(buffer, *row_num, max_line_num_len + 1);
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
}

#[derive(Clone, Copy, Debug)]
enum DisplaySuggestion {
    Underline,
    Diff,
    None,
    Add,
}

impl FileWithAnnotatedLines {
    /// Preprocess all the annotations so that they are grouped by file and by line number
    /// This helps us quickly iterate over the whole message (including secondary file spans)
    pub fn collect_annotations(
        emitter: &dyn Emitter,
        args: &FluentArgs<'_>,
        msp: &MultiSpan,
    ) -> Vec<FileWithAnnotatedLines> {
        fn add_annotation_to_file(
            file_vec: &mut Vec<FileWithAnnotatedLines>,
            file: Lrc<SourceFile>,
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

        if let Some(ref sm) = emitter.source_map() {
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
                    emitter.translate_message(m, args).map_err(Report::new).unwrap().to_string()
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
                add_annotation_to_file(&mut output, file.clone(), ann.line_start, ann.as_start());
                // 4 is the minimum vertical length of a multiline span when presented: two lines
                // of code and two lines of underline. This is not true for the special case where
                // the beginning doesn't have an underline, but the current logic seems to be
                // working correctly.
                let middle = min(ann.line_start + 4, ann.line_end);
                for line in ann.line_start + 1..middle {
                    // Every `|` that joins the beginning of the span (`___^`) to the end (`|__^`).
                    add_annotation_to_file(&mut output, file.clone(), line, ann.as_line());
                }
                let line_end = ann.line_end - 1;
                if middle < line_end {
                    add_annotation_to_file(&mut output, file.clone(), line_end, ann.as_line());
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
const OUTPUT_REPLACEMENTS: &[(char, &str)] = &[
    ('\t', "    "),   // We do our own tab replacement
    ('\u{200D}', ""), // Replace ZWJ with nothing for consistent terminal output of grapheme clusters.
    ('\u{202A}', ""), // The following unicode text flow control characters are inconsistently
    ('\u{202B}', ""), // supported across CLIs and can cause confusion due to the bytes on disk
    ('\u{202D}', ""), // not corresponding to the visible source code, so we replace them always.
    ('\u{202E}', ""),
    ('\u{2066}', ""),
    ('\u{2067}', ""),
    ('\u{2068}', ""),
    ('\u{202C}', ""),
    ('\u{2069}', ""),
];

fn normalize_whitespace(str: &str) -> String {
    let mut s = str.to_string();
    for (c, replacement) in OUTPUT_REPLACEMENTS {
        s = s.replace(*c, replacement);
    }
    s
}

fn draw_col_separator(buffer: &mut StyledBuffer, line: usize, col: usize) {
    buffer.puts(line, col, "| ", Style::LineNumber);
}

fn draw_col_separator_no_space(buffer: &mut StyledBuffer, line: usize, col: usize) {
    draw_col_separator_no_space_with_style(buffer, line, col, Style::LineNumber);
}

fn draw_col_separator_no_space_with_style(
    buffer: &mut StyledBuffer,
    line: usize,
    col: usize,
    style: Style,
) {
    buffer.putc(line, col, '|', style);
}

fn draw_range(
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

fn draw_note_separator(buffer: &mut StyledBuffer, line: usize, col: usize) {
    buffer.puts(line, col, "= ", Style::LineNumber);
}

fn draw_multiline_line(
    buffer: &mut StyledBuffer,
    line: usize,
    offset: usize,
    depth: usize,
    style: Style,
) {
    buffer.putc(line, offset + depth - 1, '|', style);
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

    let mut dst = dst.writable();

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
            dst.apply_style(*lvl, part.style)?;
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

pub enum Destination {
    Terminal(StandardStream),
    Buffered(BufferWriter),
    // The bool denotes whether we should be emitting ansi color codes or not
    Raw(Box<(dyn Write + Send)>, bool),
}

pub enum WritableDst<'a> {
    Terminal(&'a mut StandardStream),
    Buffered(&'a mut BufferWriter, Buffer),
    Raw(&'a mut (dyn Write + Send)),
    ColoredRaw(Ansi<&'a mut (dyn Write + Send)>),
}

impl Destination {
    fn from_stderr(color: ColorConfig) -> Destination {
        let choice = color.to_color_choice();
        // On Windows we'll be performing global synchronization on the entire
        // system for emitting rustc errors, so there's no need to buffer
        // anything.
        //
        // On non-Windows we rely on the atomicity of `write` to ensure errors
        // don't get all jumbled up.
        if cfg!(windows) {
            Terminal(StandardStream::stderr(choice))
        } else {
            Buffered(BufferWriter::stderr(choice))
        }
    }

    fn writable(&mut self) -> WritableDst<'_> {
        match *self {
            Destination::Terminal(ref mut t) => WritableDst::Terminal(t),
            Destination::Buffered(ref mut t) => {
                let buf = t.buffer();
                WritableDst::Buffered(t, buf)
            }
            Destination::Raw(ref mut t, false) => WritableDst::Raw(t),
            Destination::Raw(ref mut t, true) => WritableDst::ColoredRaw(Ansi::new(t)),
        }
    }

    fn supports_color(&self) -> bool {
        match *self {
            Self::Terminal(ref stream) => stream.supports_color(),
            Self::Buffered(ref buffer) => buffer.buffer().supports_color(),
            Self::Raw(_, supports_color) => supports_color,
        }
    }
}

impl<'a> WritableDst<'a> {
    fn apply_style(&mut self, lvl: Level, style: Style) -> io::Result<()> {
        let mut spec = ColorSpec::new();
        match style {
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
                if cfg!(windows) {
                    spec.set_fg(Some(Color::Cyan));
                } else {
                    spec.set_fg(Some(Color::Blue));
                }
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
                if cfg!(windows) {
                    spec.set_fg(Some(Color::Cyan));
                } else {
                    spec.set_fg(Some(Color::Blue));
                }
            }
            Style::HeaderMsg | Style::NoStyle => {}
            Style::Level(lvl) => {
                spec = lvl.color();
                spec.set_bold(true);
            }
            Style::Highlight => {
                spec.set_bold(true);
            }
        }
        self.set_color(&spec)
    }

    fn set_color(&mut self, color: &ColorSpec) -> io::Result<()> {
        match *self {
            WritableDst::Terminal(ref mut t) => t.set_color(color),
            WritableDst::Buffered(_, ref mut t) => t.set_color(color),
            WritableDst::ColoredRaw(ref mut t) => t.set_color(color),
            WritableDst::Raw(_) => Ok(()),
        }
    }

    fn reset(&mut self) -> io::Result<()> {
        match *self {
            WritableDst::Terminal(ref mut t) => t.reset(),
            WritableDst::Buffered(_, ref mut t) => t.reset(),
            WritableDst::ColoredRaw(ref mut t) => t.reset(),
            WritableDst::Raw(_) => Ok(()),
        }
    }
}

impl<'a> Write for WritableDst<'a> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match *self {
            WritableDst::Terminal(ref mut t) => t.write(bytes),
            WritableDst::Buffered(_, ref mut buf) => buf.write(bytes),
            WritableDst::Raw(ref mut w) => w.write(bytes),
            WritableDst::ColoredRaw(ref mut t) => t.write(bytes),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match *self {
            WritableDst::Terminal(ref mut t) => t.flush(),
            WritableDst::Buffered(_, ref mut buf) => buf.flush(),
            WritableDst::Raw(ref mut w) => w.flush(),
            WritableDst::ColoredRaw(ref mut w) => w.flush(),
        }
    }
}

impl<'a> Drop for WritableDst<'a> {
    fn drop(&mut self) {
        if let WritableDst::Buffered(ref mut dst, ref mut buf) = self {
            drop(dst.print(buf));
        }
    }
}

/// Whether the original and suggested code are visually similar enough to warrant extra wording.
pub fn is_case_difference(sm: &SourceMap, suggested: &str, sp: Span) -> bool {
    // FIXME: this should probably be extended to also account for `FO0` → `FOO` and unicode.
    let found = match sm.span_to_snippet(sp) {
        Ok(snippet) => snippet,
        Err(e) => {
            warn!(error = ?e, "Invalid span {:?}", sp);
            return false;
        }
    };
    let ascii_confusables = &['c', 'f', 'i', 'k', 'o', 's', 'u', 'v', 'w', 'x', 'y', 'z'];
    // All the chars that differ in capitalization are confusable (above):
    let confusable = iter::zip(found.chars(), suggested.chars())
        .filter(|(f, s)| f != s)
        .all(|(f, s)| (ascii_confusables.contains(&f) || ascii_confusables.contains(&s)));
    confusable && found.to_lowercase() == suggested.to_lowercase()
            // FIXME: We sometimes suggest the same thing we already have, which is a
            //        bug, but be defensive against that here.
            && found != suggested
}
