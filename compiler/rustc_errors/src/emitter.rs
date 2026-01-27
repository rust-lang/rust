//! The current rustc diagnostics emitter.
//!
//! An `Emitter` takes care of generating the output from a `Diag` struct.
//!
//! There are various `Emitter` implementations that generate different output formats such as
//! JSON and human readable output.
//!
//! The output types are defined in `rustc_session::config::ErrorOutputType`.

use std::borrow::Cow;
use std::error::Report;
use std::io::prelude::*;
use std::io::{self, IsTerminal};
use std::iter;
use std::path::Path;

use anstream::{AutoStream, ColorChoice};
use anstyle::{AnsiColor, Effects};
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::sync::DynSend;
use rustc_error_messages::FluentArgs;
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::source_map::SourceMap;
use rustc_span::{FileName, SourceFile, Span};
use tracing::{debug, warn};

use crate::registry::Registry;
use crate::timings::TimingRecord;
use crate::translation::Translator;
use crate::{
    CodeSuggestion, DiagInner, DiagMessage, Level, MultiSpan, Style, Subdiag, SuggestionStyle,
};

/// Describes the way the content of the `rendered` field of the json output is generated
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HumanReadableErrorType {
    pub short: bool,
    pub unicode: bool,
}

impl HumanReadableErrorType {
    pub fn short(&self) -> bool {
        self.short
    }
}

pub enum TimingEvent {
    Start,
    End,
}

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
                    let mut span = sp;
                    while let Some(callsite) = span.parent_callsite() {
                        span = callsite;
                        if !source_map.is_imported(span) {
                            return Some((sp, span));
                        }
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

/// An emitter that adds a note to each diagnostic.
pub struct EmitterWithNote {
    pub emitter: Box<dyn Emitter + DynSend>,
    pub note: String,
}

impl Emitter for EmitterWithNote {
    fn source_map(&self) -> Option<&SourceMap> {
        None
    }

    fn emit_diagnostic(&mut self, mut diag: DiagInner, registry: &Registry) {
        diag.sub(Level::Note, self.note.clone(), MultiSpan::new());
        self.emitter.emit_diagnostic(diag, registry);
    }

    fn translator(&self) -> &Translator {
        self.emitter.translator()
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

pub(crate) fn normalize_whitespace(s: &str) -> String {
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

pub type Destination = AutoStream<Box<dyn Write + Send>>;

struct Buffy {
    buffer_writer: std::io::Stderr,
    buffer: Vec<u8>,
}

impl Write for Buffy {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buffer_writer.write_all(&self.buffer)?;
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

pub fn stderr_destination(color: ColorConfig) -> Destination {
    let buffer_writer = std::io::stderr();
    // We need to resolve `ColorChoice::Auto` before `Box`ing since
    // `ColorChoice::Auto` on `dyn Write` will always resolve to `Never`
    let choice = get_stderr_color_choice(color, &buffer_writer);
    // On Windows we'll be performing global synchronization on the entire
    // system for emitting rustc errors, so there's no need to buffer
    // anything.
    //
    // On non-Windows we rely on the atomicity of `write` to ensure errors
    // don't get all jumbled up.
    if cfg!(windows) {
        AutoStream::new(Box::new(buffer_writer), choice)
    } else {
        let buffer = Vec::new();
        AutoStream::new(Box::new(Buffy { buffer_writer, buffer }), choice)
    }
}

pub fn get_stderr_color_choice(color: ColorConfig, stderr: &std::io::Stderr) -> ColorChoice {
    let choice = color.to_color_choice();
    if matches!(choice, ColorChoice::Auto) { AutoStream::choice(stderr) } else { choice }
}

/// On Windows, BRIGHT_BLUE is hard to read on black. Use cyan instead.
///
/// See #36178.
const BRIGHT_BLUE: anstyle::Style = if cfg!(windows) {
    AnsiColor::BrightCyan.on_default()
} else {
    AnsiColor::BrightBlue.on_default()
};

impl Style {
    pub(crate) fn anstyle(&self, lvl: Level) -> anstyle::Style {
        match self {
            Style::Addition => AnsiColor::BrightGreen.on_default(),
            Style::Removal => AnsiColor::BrightRed.on_default(),
            Style::LineAndColumn => anstyle::Style::new(),
            Style::LineNumber => BRIGHT_BLUE.effects(Effects::BOLD),
            Style::Quotation => anstyle::Style::new(),
            Style::MainHeaderMsg => if cfg!(windows) {
                AnsiColor::BrightWhite.on_default()
            } else {
                anstyle::Style::new()
            }
            .effects(Effects::BOLD),
            Style::UnderlinePrimary | Style::LabelPrimary => lvl.color().effects(Effects::BOLD),
            Style::UnderlineSecondary | Style::LabelSecondary => BRIGHT_BLUE.effects(Effects::BOLD),
            Style::HeaderMsg | Style::NoStyle => anstyle::Style::new(),
            Style::Level(lvl) => lvl.color().effects(Effects::BOLD),
            Style::Highlight => AnsiColor::Magenta.on_default().effects(Effects::BOLD),
        }
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

        // Letters whose lowercase version is very similar to the uppercase
        // version.
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
