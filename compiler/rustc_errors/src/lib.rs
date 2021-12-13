//! Diagnostics creation and emission for `rustc`.
//!
//! This module contains the code for creating and emitting diagnostics.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(crate_visibility_modifier)]
#![feature(backtrace)]
#![feature(if_let_guard)]
#![feature(let_else)]
#![feature(nll)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

pub use emitter::ColorConfig;

use Level::*;

use emitter::{is_case_difference, Emitter, EmitterWriter};
use registry::Registry;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{self, Lock, Lrc};
use rustc_data_structures::AtomicRef;
pub use rustc_lint_defs::{pluralize, Applicability};
use rustc_serialize::json::Json;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::source_map::SourceMap;
use rustc_span::{Loc, MultiSpan, Span};

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::panic;
use std::path::Path;
use std::{error, fmt};

use termcolor::{Color, ColorSpec};

pub mod annotate_snippet_emitter_writer;
mod diagnostic;
mod diagnostic_builder;
pub mod emitter;
pub mod json;
mod lock;
pub mod registry;
mod snippet;
mod styled_buffer;
pub use snippet::Style;

pub type PResult<'a, T> = Result<T, DiagnosticBuilder<'a>>;

// `PResult` is used a lot. Make sure it doesn't unintentionally get bigger.
// (See also the comment on `DiagnosticBuilderInner`.)
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(PResult<'_, bool>, 16);

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Encodable, Decodable)]
pub enum SuggestionStyle {
    /// Hide the suggested code when displaying this suggestion inline.
    HideCodeInline,
    /// Always hide the suggested code but display the message.
    HideCodeAlways,
    /// Do not display this suggestion in the cli output, it is only meant for tools.
    CompletelyHidden,
    /// Always show the suggested code.
    /// This will *not* show the code if the suggestion is inline *and* the suggested code is
    /// empty.
    ShowCode,
    /// Always show the suggested code independently.
    ShowAlways,
}

impl SuggestionStyle {
    fn hide_inline(&self) -> bool {
        !matches!(*self, SuggestionStyle::ShowCode)
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct ToolMetadata(pub Option<Json>);

impl ToolMetadata {
    fn new(json: Json) -> Self {
        ToolMetadata(Some(json))
    }

    fn is_set(&self) -> bool {
        self.0.is_some()
    }
}

impl Hash for ToolMetadata {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

// Doesn't really need to round-trip
impl<D: Decoder> Decodable<D> for ToolMetadata {
    fn decode(_d: &mut D) -> Result<Self, D::Error> {
        Ok(ToolMetadata(None))
    }
}

impl<S: Encoder> Encodable<S> for ToolMetadata {
    fn encode(&self, e: &mut S) -> Result<(), S::Error> {
        match &self.0 {
            None => e.emit_unit(),
            Some(json) => json.encode(e),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub struct CodeSuggestion {
    /// Each substitute can have multiple variants due to multiple
    /// applicable suggestions
    ///
    /// `foo.bar` might be replaced with `a.b` or `x.y` by replacing
    /// `foo` and `bar` on their own:
    ///
    /// ```
    /// vec![
    ///     Substitution { parts: vec![(0..3, "a"), (4..7, "b")] },
    ///     Substitution { parts: vec![(0..3, "x"), (4..7, "y")] },
    /// ]
    /// ```
    ///
    /// or by replacing the entire span:
    ///
    /// ```
    /// vec![
    ///     Substitution { parts: vec![(0..7, "a.b")] },
    ///     Substitution { parts: vec![(0..7, "x.y")] },
    /// ]
    /// ```
    pub substitutions: Vec<Substitution>,
    pub msg: String,
    /// Visual representation of this suggestion.
    pub style: SuggestionStyle,
    /// Whether or not the suggestion is approximate
    ///
    /// Sometimes we may show suggestions with placeholders,
    /// which are useful for users but not useful for
    /// tools like rustfix
    pub applicability: Applicability,
    /// Tool-specific metadata
    pub tool_metadata: ToolMetadata,
}

#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
/// See the docs on `CodeSuggestion::substitutions`
pub struct Substitution {
    pub parts: Vec<SubstitutionPart>,
}

#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub struct SubstitutionPart {
    pub span: Span,
    pub snippet: String,
}

/// Used to translate between `Span`s and byte positions within a single output line in highlighted
/// code of structured suggestions.
#[derive(Debug, Clone, Copy)]
pub struct SubstitutionHighlight {
    start: usize,
    end: usize,
}

impl SubstitutionPart {
    pub fn is_addition(&self, sm: &SourceMap) -> bool {
        !self.snippet.is_empty()
            && sm
                .span_to_snippet(self.span)
                .map_or(self.span.is_empty(), |snippet| snippet.trim().is_empty())
    }

    pub fn is_deletion(&self) -> bool {
        self.snippet.trim().is_empty()
    }

    pub fn is_replacement(&self, sm: &SourceMap) -> bool {
        !self.snippet.is_empty()
            && sm
                .span_to_snippet(self.span)
                .map_or(!self.span.is_empty(), |snippet| !snippet.trim().is_empty())
    }
}

impl CodeSuggestion {
    /// Returns the assembled code suggestions, whether they should be shown with an underline
    /// and whether the substitution only differs in capitalization.
    pub fn splice_lines(
        &self,
        sm: &SourceMap,
    ) -> Vec<(String, Vec<SubstitutionPart>, Vec<Vec<SubstitutionHighlight>>, bool)> {
        // For the `Vec<Vec<SubstitutionHighlight>>` value, the first level of the vector
        // corresponds to the output snippet's lines, while the second level corresponds to the
        // substrings within that line that should be highlighted.

        use rustc_span::{CharPos, Pos};

        /// Append to a buffer the remainder of the line of existing source code, and return the
        /// count of lines that have been added for accurate highlighting.
        fn push_trailing(
            buf: &mut String,
            line_opt: Option<&Cow<'_, str>>,
            lo: &Loc,
            hi_opt: Option<&Loc>,
        ) -> usize {
            let mut line_count = 0;
            let (lo, hi_opt) = (lo.col.to_usize(), hi_opt.map(|hi| hi.col.to_usize()));
            if let Some(line) = line_opt {
                if let Some(lo) = line.char_indices().map(|(i, _)| i).nth(lo) {
                    let hi_opt = hi_opt.and_then(|hi| line.char_indices().map(|(i, _)| i).nth(hi));
                    match hi_opt {
                        Some(hi) if hi > lo => {
                            line_count = line[lo..hi].matches('\n').count();
                            buf.push_str(&line[lo..hi])
                        }
                        Some(_) => (),
                        None => {
                            line_count = line[lo..].matches('\n').count();
                            buf.push_str(&line[lo..])
                        }
                    }
                }
                if hi_opt.is_none() {
                    buf.push('\n');
                }
            }
            line_count
        }

        assert!(!self.substitutions.is_empty());

        self.substitutions
            .iter()
            .filter(|subst| {
                // Suggestions coming from macros can have malformed spans. This is a heavy
                // handed approach to avoid ICEs by ignoring the suggestion outright.
                let invalid = subst.parts.iter().any(|item| sm.is_valid_span(item.span).is_err());
                if invalid {
                    debug!("splice_lines: suggestion contains an invalid span: {:?}", subst);
                }
                !invalid
            })
            .cloned()
            .filter_map(|mut substitution| {
                // Assumption: all spans are in the same file, and all spans
                // are disjoint. Sort in ascending order.
                substitution.parts.sort_by_key(|part| part.span.lo());

                // Find the bounding span.
                let lo = substitution.parts.iter().map(|part| part.span.lo()).min()?;
                let hi = substitution.parts.iter().map(|part| part.span.hi()).max()?;
                let bounding_span = Span::with_root_ctxt(lo, hi);
                // The different spans might belong to different contexts, if so ignore suggestion.
                let lines = sm.span_to_lines(bounding_span).ok()?;
                assert!(!lines.lines.is_empty() || bounding_span.is_dummy());

                // We can't splice anything if the source is unavailable.
                if !sm.ensure_source_file_source_present(lines.file.clone()) {
                    return None;
                }

                let mut highlights = vec![];
                // To build up the result, we do this for each span:
                // - push the line segment trailing the previous span
                //   (at the beginning a "phantom" span pointing at the start of the line)
                // - push lines between the previous and current span (if any)
                // - if the previous and current span are not on the same line
                //   push the line segment leading up to the current span
                // - splice in the span substitution
                //
                // Finally push the trailing line segment of the last span
                let sf = &lines.file;
                let mut prev_hi = sm.lookup_char_pos(bounding_span.lo());
                prev_hi.col = CharPos::from_usize(0);
                let mut prev_line =
                    lines.lines.get(0).and_then(|line0| sf.get_line(line0.line_index));
                let mut buf = String::new();

                let mut line_highlight = vec![];
                // We need to keep track of the difference between the existing code and the added
                // or deleted code in order to point at the correct column *after* substitution.
                let mut acc = 0;
                for part in &substitution.parts {
                    let cur_lo = sm.lookup_char_pos(part.span.lo());
                    if prev_hi.line == cur_lo.line {
                        let mut count =
                            push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, Some(&cur_lo));
                        while count > 0 {
                            highlights.push(std::mem::take(&mut line_highlight));
                            acc = 0;
                            count -= 1;
                        }
                    } else {
                        acc = 0;
                        highlights.push(std::mem::take(&mut line_highlight));
                        let mut count = push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, None);
                        while count > 0 {
                            highlights.push(std::mem::take(&mut line_highlight));
                            count -= 1;
                        }
                        // push lines between the previous and current span (if any)
                        for idx in prev_hi.line..(cur_lo.line - 1) {
                            if let Some(line) = sf.get_line(idx) {
                                buf.push_str(line.as_ref());
                                buf.push('\n');
                                highlights.push(std::mem::take(&mut line_highlight));
                            }
                        }
                        if let Some(cur_line) = sf.get_line(cur_lo.line - 1) {
                            let end = match cur_line.char_indices().nth(cur_lo.col.to_usize()) {
                                Some((i, _)) => i,
                                None => cur_line.len(),
                            };
                            buf.push_str(&cur_line[..end]);
                        }
                    }
                    // Add a whole line highlight per line in the snippet.
                    let len: isize = part
                        .snippet
                        .split('\n')
                        .next()
                        .unwrap_or(&part.snippet)
                        .chars()
                        .map(|c| match c {
                            '\t' => 4,
                            _ => 1,
                        })
                        .sum();
                    line_highlight.push(SubstitutionHighlight {
                        start: (cur_lo.col.0 as isize + acc) as usize,
                        end: (cur_lo.col.0 as isize + acc + len) as usize,
                    });
                    buf.push_str(&part.snippet);
                    let cur_hi = sm.lookup_char_pos(part.span.hi());
                    if prev_hi.line == cur_lo.line && cur_hi.line == cur_lo.line {
                        // Account for the difference between the width of the current code and the
                        // snippet being suggested, so that the *later* suggestions are correctly
                        // aligned on the screen.
                        acc += len as isize - (cur_hi.col.0 - cur_lo.col.0) as isize;
                    }
                    prev_hi = cur_hi;
                    prev_line = sf.get_line(prev_hi.line - 1);
                    for line in part.snippet.split('\n').skip(1) {
                        acc = 0;
                        highlights.push(std::mem::take(&mut line_highlight));
                        let end: usize = line
                            .chars()
                            .map(|c| match c {
                                '\t' => 4,
                                _ => 1,
                            })
                            .sum();
                        line_highlight.push(SubstitutionHighlight { start: 0, end });
                    }
                }
                highlights.push(std::mem::take(&mut line_highlight));
                let only_capitalization = is_case_difference(sm, &buf, bounding_span);
                // if the replacement already ends with a newline, don't print the next line
                if !buf.ends_with('\n') {
                    push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, None);
                }
                // remove trailing newlines
                while buf.ends_with('\n') {
                    buf.pop();
                }
                Some((buf, substitution.parts, highlights, only_capitalization))
            })
            .collect()
    }
}

pub use rustc_span::fatal_error::{FatalError, FatalErrorMarker};

/// Signifies that the compiler died with an explicit call to `.bug`
/// or `.span_bug` rather than a failed assertion, etc.
#[derive(Copy, Clone, Debug)]
pub struct ExplicitBug;

impl fmt::Display for ExplicitBug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parser internal bug")
    }
}

impl error::Error for ExplicitBug {}

pub use diagnostic::{Diagnostic, DiagnosticId, DiagnosticStyledString, SubDiagnostic};
pub use diagnostic_builder::DiagnosticBuilder;
use std::backtrace::Backtrace;

/// A handler deals with errors and other compiler output.
/// Certain errors (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct Handler {
    flags: HandlerFlags,
    inner: Lock<HandlerInner>,
}

/// This inner struct exists to keep it all behind a single lock;
/// this is done to prevent possible deadlocks in a multi-threaded compiler,
/// as well as inconsistent state observation.
struct HandlerInner {
    flags: HandlerFlags,
    /// The number of lint errors that have been emitted.
    lint_err_count: usize,
    /// The number of errors that have been emitted, including duplicates.
    ///
    /// This is not necessarily the count that's reported to the user once
    /// compilation ends.
    err_count: usize,
    warn_count: usize,
    deduplicated_err_count: usize,
    emitter: Box<dyn Emitter + sync::Send>,
    delayed_span_bugs: Vec<Diagnostic>,
    delayed_good_path_bugs: Vec<DelayedDiagnostic>,

    /// This set contains the `DiagnosticId` of all emitted diagnostics to avoid
    /// emitting the same diagnostic with extended help (`--teach`) twice, which
    /// would be unnecessary repetition.
    taught_diagnostics: FxHashSet<DiagnosticId>,

    /// Used to suggest rustc --explain <error code>
    emitted_diagnostic_codes: FxHashSet<DiagnosticId>,

    /// This set contains a hash of every diagnostic that has been emitted by
    /// this handler. These hashes is used to avoid emitting the same error
    /// twice.
    emitted_diagnostics: FxHashSet<u128>,

    /// Stashed diagnostics emitted in one stage of the compiler that may be
    /// stolen by other stages (e.g. to improve them and add more information).
    /// The stashed diagnostics count towards the total error count.
    /// When `.abort_if_errors()` is called, these are also emitted.
    stashed_diagnostics: FxIndexMap<(Span, StashKey), Diagnostic>,

    /// The warning count, used for a recap upon finishing
    deduplicated_warn_count: usize,

    future_breakage_diagnostics: Vec<Diagnostic>,

    /// If set to `true`, no warning or error will be emitted.
    quiet: bool,
}

/// A key denoting where from a diagnostic was stashed.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum StashKey {
    ItemNoType,
}

fn default_track_diagnostic(_: &Diagnostic) {}

pub static TRACK_DIAGNOSTICS: AtomicRef<fn(&Diagnostic)> =
    AtomicRef::new(&(default_track_diagnostic as fn(&_)));

#[derive(Copy, Clone, Default)]
pub struct HandlerFlags {
    /// If false, warning-level lints are suppressed.
    /// (rustc: see `--allow warnings` and `--cap-lints`)
    pub can_emit_warnings: bool,
    /// If true, error-level diagnostics are upgraded to bug-level.
    /// (rustc: see `-Z treat-err-as-bug`)
    pub treat_err_as_bug: Option<NonZeroUsize>,
    /// If true, immediately emit diagnostics that would otherwise be buffered.
    /// (rustc: see `-Z dont-buffer-diagnostics` and `-Z treat-err-as-bug`)
    pub dont_buffer_diagnostics: bool,
    /// If true, immediately print bugs registered with `delay_span_bug`.
    /// (rustc: see `-Z report-delayed-bugs`)
    pub report_delayed_bugs: bool,
    /// Show macro backtraces.
    /// (rustc: see `-Z macro-backtrace`)
    pub macro_backtrace: bool,
    /// If true, identical diagnostics are reported only once.
    pub deduplicate_diagnostics: bool,
}

impl Drop for HandlerInner {
    fn drop(&mut self) {
        self.emit_stashed_diagnostics();

        if !self.has_errors() {
            let bugs = std::mem::replace(&mut self.delayed_span_bugs, Vec::new());
            self.flush_delayed(bugs, "no errors encountered even though `delay_span_bug` issued");
        }

        if !self.has_any_message() {
            let bugs = std::mem::replace(&mut self.delayed_good_path_bugs, Vec::new());
            self.flush_delayed(
                bugs.into_iter().map(DelayedDiagnostic::decorate).collect(),
                "no warnings or errors encountered even though `delayed_good_path_bugs` issued",
            );
        }
    }
}

impl Handler {
    pub fn with_tty_emitter(
        color_config: ColorConfig,
        can_emit_warnings: bool,
        treat_err_as_bug: Option<NonZeroUsize>,
        sm: Option<Lrc<SourceMap>>,
    ) -> Self {
        Self::with_tty_emitter_and_flags(
            color_config,
            sm,
            HandlerFlags { can_emit_warnings, treat_err_as_bug, ..Default::default() },
        )
    }

    pub fn with_tty_emitter_and_flags(
        color_config: ColorConfig,
        sm: Option<Lrc<SourceMap>>,
        flags: HandlerFlags,
    ) -> Self {
        let emitter = Box::new(EmitterWriter::stderr(
            color_config,
            sm,
            false,
            false,
            None,
            flags.macro_backtrace,
        ));
        Self::with_emitter_and_flags(emitter, flags)
    }

    pub fn with_emitter(
        can_emit_warnings: bool,
        treat_err_as_bug: Option<NonZeroUsize>,
        emitter: Box<dyn Emitter + sync::Send>,
    ) -> Self {
        Handler::with_emitter_and_flags(
            emitter,
            HandlerFlags { can_emit_warnings, treat_err_as_bug, ..Default::default() },
        )
    }

    pub fn with_emitter_and_flags(
        emitter: Box<dyn Emitter + sync::Send>,
        flags: HandlerFlags,
    ) -> Self {
        Self {
            flags,
            inner: Lock::new(HandlerInner {
                flags,
                lint_err_count: 0,
                err_count: 0,
                warn_count: 0,
                deduplicated_err_count: 0,
                deduplicated_warn_count: 0,
                emitter,
                delayed_span_bugs: Vec::new(),
                delayed_good_path_bugs: Vec::new(),
                taught_diagnostics: Default::default(),
                emitted_diagnostic_codes: Default::default(),
                emitted_diagnostics: Default::default(),
                stashed_diagnostics: Default::default(),
                future_breakage_diagnostics: Vec::new(),
                quiet: false,
            }),
        }
    }

    pub fn with_disabled_diagnostic<T, F: FnOnce() -> T>(&self, f: F) -> T {
        let prev = self.inner.borrow_mut().quiet;
        self.inner.borrow_mut().quiet = true;
        let ret = f();
        self.inner.borrow_mut().quiet = prev;
        ret
    }

    // This is here to not allow mutation of flags;
    // as of this writing it's only used in tests in librustc_middle.
    pub fn can_emit_warnings(&self) -> bool {
        self.flags.can_emit_warnings
    }

    /// Resets the diagnostic error count as well as the cached emitted diagnostics.
    ///
    /// NOTE: *do not* call this function from rustc. It is only meant to be called from external
    /// tools that want to reuse a `Parser` cleaning the previously emitted diagnostics as well as
    /// the overall count of emitted error diagnostics.
    pub fn reset_err_count(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.err_count = 0;
        inner.warn_count = 0;
        inner.deduplicated_err_count = 0;
        inner.deduplicated_warn_count = 0;

        // actually free the underlying memory (which `clear` would not do)
        inner.delayed_span_bugs = Default::default();
        inner.delayed_good_path_bugs = Default::default();
        inner.taught_diagnostics = Default::default();
        inner.emitted_diagnostic_codes = Default::default();
        inner.emitted_diagnostics = Default::default();
        inner.stashed_diagnostics = Default::default();
    }

    /// Stash a given diagnostic with the given `Span` and `StashKey` as the key for later stealing.
    pub fn stash_diagnostic(&self, span: Span, key: StashKey, diag: Diagnostic) {
        let mut inner = self.inner.borrow_mut();
        // FIXME(Centril, #69537): Consider reintroducing panic on overwriting a stashed diagnostic
        // if/when we have a more robust macro-friendly replacement for `(span, key)` as a key.
        // See the PR for a discussion.
        inner.stashed_diagnostics.insert((span, key), diag);
    }

    /// Steal a previously stashed diagnostic with the given `Span` and `StashKey` as the key.
    pub fn steal_diagnostic(&self, span: Span, key: StashKey) -> Option<DiagnosticBuilder<'_>> {
        self.inner
            .borrow_mut()
            .stashed_diagnostics
            .remove(&(span, key))
            .map(|diag| DiagnosticBuilder::new_diagnostic(self, diag))
    }

    /// Emit all stashed diagnostics.
    pub fn emit_stashed_diagnostics(&self) {
        self.inner.borrow_mut().emit_stashed_diagnostics();
    }

    /// Construct a dummy builder with `Level::Cancelled`.
    ///
    /// Using this will neither report anything to the user (e.g. a warning),
    /// nor will compilation cancel as a result.
    pub fn struct_dummy(&self) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Cancelled, "")
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    ///
    /// The builder will be canceled if warnings cannot be emitted.
    pub fn struct_span_warn(&self, span: impl Into<MultiSpan>, msg: &str) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_warn(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    ///
    /// This will "force" the warning meaning it will not be canceled even
    /// if warnings cannot be emitted.
    pub fn struct_span_force_warn(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
    ) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_force_warn(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Allow` level at the given `span` and with the `msg`.
    pub fn struct_span_allow(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
    ) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_allow(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    /// Also include a code.
    pub fn struct_span_warn_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_span_warn(span, msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Warning` level with the `msg`.
    ///
    /// The builder will be canceled if warnings cannot be emitted.
    pub fn struct_warn(&self, msg: &str) -> DiagnosticBuilder<'_> {
        let mut result = DiagnosticBuilder::new(self, Level::Warning, msg);
        if !self.flags.can_emit_warnings {
            result.cancel();
        }
        result
    }

    /// Construct a builder at the `Warning` level with the `msg`.
    ///
    /// This will "force" a warning meaning it will not be canceled even
    /// if warnings cannot be emitted.
    pub fn struct_force_warn(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Warning, msg)
    }

    /// Construct a builder at the `Allow` level with the `msg`.
    pub fn struct_allow(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Allow, msg)
    }

    /// Construct a builder at the `Error` level at the given `span` and with the `msg`.
    pub fn struct_span_err(&self, span: impl Into<MultiSpan>, msg: &str) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_err(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Error` level at the given `span`, with the `msg`, and `code`.
    pub fn struct_span_err_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_span_err(span, msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Error` level with the `msg`.
    // FIXME: This method should be removed (every error should have an associated error code).
    pub fn struct_err(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Error { lint: false }, msg)
    }

    /// This should only be used by `rustc_middle::lint::struct_lint_level`. Do not use it for hard errors.
    #[doc(hidden)]
    pub fn struct_err_lint(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Error { lint: true }, msg)
    }

    /// Construct a builder at the `Error` level with the `msg` and the `code`.
    pub fn struct_err_with_code(&self, msg: &str, code: DiagnosticId) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_err(msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Fatal` level at the given `span` and with the `msg`.
    pub fn struct_span_fatal(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
    ) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_fatal(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Fatal` level at the given `span`, with the `msg`, and `code`.
    pub fn struct_span_fatal_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_> {
        let mut result = self.struct_span_fatal(span, msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Error` level with the `msg`.
    pub fn struct_fatal(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Fatal, msg)
    }

    /// Construct a builder at the `Help` level with the `msg`.
    pub fn struct_help(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Help, msg)
    }

    /// Construct a builder at the `Note` level with the `msg`.
    pub fn struct_note_without_error(&self, msg: &str) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Level::Note, msg)
    }

    pub fn span_fatal(&self, span: impl Into<MultiSpan>, msg: &str) -> ! {
        self.emit_diag_at_span(Diagnostic::new(Fatal, msg), span);
        FatalError.raise()
    }

    pub fn span_fatal_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: &str,
        code: DiagnosticId,
    ) -> ! {
        self.emit_diag_at_span(Diagnostic::new_with_code(Fatal, Some(code), msg), span);
        FatalError.raise()
    }

    pub fn span_err(&self, span: impl Into<MultiSpan>, msg: &str) {
        self.emit_diag_at_span(Diagnostic::new(Error { lint: false }, msg), span);
    }

    pub fn span_err_with_code(&self, span: impl Into<MultiSpan>, msg: &str, code: DiagnosticId) {
        self.emit_diag_at_span(
            Diagnostic::new_with_code(Error { lint: false }, Some(code), msg),
            span,
        );
    }

    pub fn span_warn(&self, span: impl Into<MultiSpan>, msg: &str) {
        self.emit_diag_at_span(Diagnostic::new(Warning, msg), span);
    }

    pub fn span_warn_with_code(&self, span: impl Into<MultiSpan>, msg: &str, code: DiagnosticId) {
        self.emit_diag_at_span(Diagnostic::new_with_code(Warning, Some(code), msg), span);
    }

    pub fn span_bug(&self, span: impl Into<MultiSpan>, msg: &str) -> ! {
        self.inner.borrow_mut().span_bug(span, msg)
    }

    #[track_caller]
    pub fn delay_span_bug(&self, span: impl Into<MultiSpan>, msg: &str) {
        self.inner.borrow_mut().delay_span_bug(span, msg)
    }

    pub fn delay_good_path_bug(&self, msg: &str) {
        self.inner.borrow_mut().delay_good_path_bug(msg)
    }

    pub fn span_bug_no_panic(&self, span: impl Into<MultiSpan>, msg: &str) {
        self.emit_diag_at_span(Diagnostic::new(Bug, msg), span);
    }

    pub fn span_note_without_error(&self, span: impl Into<MultiSpan>, msg: &str) {
        self.emit_diag_at_span(Diagnostic::new(Note, msg), span);
    }

    pub fn span_note_diag(&self, span: Span, msg: &str) -> DiagnosticBuilder<'_> {
        let mut db = DiagnosticBuilder::new(self, Note, msg);
        db.set_span(span);
        db
    }

    // NOTE: intentionally doesn't raise an error so rustc_codegen_ssa only reports fatal errors in the main thread
    pub fn fatal(&self, msg: &str) -> FatalError {
        self.inner.borrow_mut().fatal(msg)
    }

    pub fn err(&self, msg: &str) {
        self.inner.borrow_mut().err(msg);
    }

    pub fn warn(&self, msg: &str) {
        let mut db = DiagnosticBuilder::new(self, Warning, msg);
        db.emit();
    }

    pub fn note_without_error(&self, msg: &str) {
        DiagnosticBuilder::new(self, Note, msg).emit();
    }

    pub fn bug(&self, msg: &str) -> ! {
        self.inner.borrow_mut().bug(msg)
    }

    #[inline]
    pub fn err_count(&self) -> usize {
        self.inner.borrow().err_count()
    }

    pub fn has_errors(&self) -> bool {
        self.inner.borrow().has_errors()
    }
    pub fn has_errors_or_lint_errors(&self) -> bool {
        self.inner.borrow().has_errors_or_lint_errors()
    }
    pub fn has_errors_or_delayed_span_bugs(&self) -> bool {
        self.inner.borrow().has_errors_or_delayed_span_bugs()
    }

    pub fn print_error_count(&self, registry: &Registry) {
        self.inner.borrow_mut().print_error_count(registry)
    }

    pub fn take_future_breakage_diagnostics(&self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.inner.borrow_mut().future_breakage_diagnostics)
    }

    pub fn abort_if_errors(&self) {
        self.inner.borrow_mut().abort_if_errors()
    }

    /// `true` if we haven't taught a diagnostic with this code already.
    /// The caller must then teach the user about such a diagnostic.
    ///
    /// Used to suppress emitting the same error multiple times with extended explanation when
    /// calling `-Zteach`.
    pub fn must_teach(&self, code: &DiagnosticId) -> bool {
        self.inner.borrow_mut().must_teach(code)
    }

    pub fn force_print_diagnostic(&self, db: Diagnostic) {
        self.inner.borrow_mut().force_print_diagnostic(db)
    }

    pub fn emit_diagnostic(&self, diagnostic: &Diagnostic) {
        self.inner.borrow_mut().emit_diagnostic(diagnostic)
    }

    fn emit_diag_at_span(&self, mut diag: Diagnostic, sp: impl Into<MultiSpan>) {
        let mut inner = self.inner.borrow_mut();
        inner.emit_diagnostic(diag.set_span(sp));
    }

    pub fn emit_artifact_notification(&self, path: &Path, artifact_type: &str) {
        self.inner.borrow_mut().emit_artifact_notification(path, artifact_type)
    }

    pub fn emit_future_breakage_report(&self, diags: Vec<Diagnostic>) {
        self.inner.borrow_mut().emitter.emit_future_breakage_report(diags)
    }

    pub fn emit_unused_externs(&self, lint_level: &str, unused_externs: &[&str]) {
        self.inner.borrow_mut().emit_unused_externs(lint_level, unused_externs)
    }

    pub fn delay_as_bug(&self, diagnostic: Diagnostic) {
        self.inner.borrow_mut().delay_as_bug(diagnostic)
    }
}

impl HandlerInner {
    fn must_teach(&mut self, code: &DiagnosticId) -> bool {
        self.taught_diagnostics.insert(code.clone())
    }

    fn force_print_diagnostic(&mut self, db: Diagnostic) {
        self.emitter.emit_diagnostic(&db);
    }

    /// Emit all stashed diagnostics.
    fn emit_stashed_diagnostics(&mut self) {
        let diags = self.stashed_diagnostics.drain(..).map(|x| x.1).collect::<Vec<_>>();
        diags.iter().for_each(|diag| self.emit_diagnostic(diag));
    }

    fn emit_diagnostic(&mut self, diagnostic: &Diagnostic) {
        if diagnostic.cancelled() || self.quiet {
            return;
        }

        if diagnostic.has_future_breakage() {
            self.future_breakage_diagnostics.push(diagnostic.clone());
        }

        if diagnostic.level == Warning
            && !self.flags.can_emit_warnings
            && !diagnostic.is_force_warn()
        {
            if diagnostic.has_future_breakage() {
                (*TRACK_DIAGNOSTICS)(diagnostic);
            }
            return;
        }

        (*TRACK_DIAGNOSTICS)(diagnostic);

        if diagnostic.level == Allow {
            return;
        }

        if let Some(ref code) = diagnostic.code {
            self.emitted_diagnostic_codes.insert(code.clone());
        }

        let already_emitted = |this: &mut Self| {
            let mut hasher = StableHasher::new();
            diagnostic.hash(&mut hasher);
            let diagnostic_hash = hasher.finish();
            !this.emitted_diagnostics.insert(diagnostic_hash)
        };

        // Only emit the diagnostic if we've been asked to deduplicate and
        // haven't already emitted an equivalent diagnostic.
        if !(self.flags.deduplicate_diagnostics && already_emitted(self)) {
            self.emitter.emit_diagnostic(diagnostic);
            if diagnostic.is_error() {
                self.deduplicated_err_count += 1;
            } else if diagnostic.level == Warning {
                self.deduplicated_warn_count += 1;
            }
        }
        if diagnostic.is_error() {
            if matches!(diagnostic.level, Level::Error { lint: true }) {
                self.bump_lint_err_count();
            } else {
                self.bump_err_count();
            }
        } else {
            self.bump_warn_count();
        }
    }

    fn emit_artifact_notification(&mut self, path: &Path, artifact_type: &str) {
        self.emitter.emit_artifact_notification(path, artifact_type);
    }

    fn emit_unused_externs(&mut self, lint_level: &str, unused_externs: &[&str]) {
        self.emitter.emit_unused_externs(lint_level, unused_externs);
    }

    fn treat_err_as_bug(&self) -> bool {
        self.flags
            .treat_err_as_bug
            .map_or(false, |c| self.err_count() + self.lint_err_count >= c.get())
    }

    fn print_error_count(&mut self, registry: &Registry) {
        self.emit_stashed_diagnostics();

        let warnings = match self.deduplicated_warn_count {
            0 => String::new(),
            1 => "1 warning emitted".to_string(),
            count => format!("{} warnings emitted", count),
        };
        let errors = match self.deduplicated_err_count {
            0 => String::new(),
            1 => "aborting due to previous error".to_string(),
            count => format!("aborting due to {} previous errors", count),
        };
        if self.treat_err_as_bug() {
            return;
        }

        match (errors.len(), warnings.len()) {
            (0, 0) => return,
            (0, _) => self.emitter.emit_diagnostic(&Diagnostic::new(Level::Warning, &warnings)),
            (_, 0) => {
                let _ = self.fatal(&errors);
            }
            (_, _) => {
                let _ = self.fatal(&format!("{}; {}", &errors, &warnings));
            }
        }

        let can_show_explain = self.emitter.should_show_explain();
        let are_there_diagnostics = !self.emitted_diagnostic_codes.is_empty();
        if can_show_explain && are_there_diagnostics {
            let mut error_codes = self
                .emitted_diagnostic_codes
                .iter()
                .filter_map(|x| match &x {
                    DiagnosticId::Error(s)
                        if registry.try_find_description(s).map_or(false, |o| o.is_some()) =>
                    {
                        Some(s.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if !error_codes.is_empty() {
                error_codes.sort();
                if error_codes.len() > 1 {
                    let limit = if error_codes.len() > 9 { 9 } else { error_codes.len() };
                    self.failure(&format!(
                        "Some errors have detailed explanations: {}{}",
                        error_codes[..limit].join(", "),
                        if error_codes.len() > 9 { "..." } else { "." }
                    ));
                    self.failure(&format!(
                        "For more information about an error, try \
                         `rustc --explain {}`.",
                        &error_codes[0]
                    ));
                } else {
                    self.failure(&format!(
                        "For more information about this error, try \
                         `rustc --explain {}`.",
                        &error_codes[0]
                    ));
                }
            }
        }
    }

    #[inline]
    fn err_count(&self) -> usize {
        self.err_count + self.stashed_diagnostics.len()
    }

    fn has_errors(&self) -> bool {
        self.err_count() > 0
    }
    fn has_errors_or_lint_errors(&self) -> bool {
        self.has_errors() || self.lint_err_count > 0
    }
    fn has_errors_or_delayed_span_bugs(&self) -> bool {
        self.has_errors() || !self.delayed_span_bugs.is_empty()
    }
    fn has_any_message(&self) -> bool {
        self.err_count() > 0 || self.lint_err_count > 0 || self.warn_count > 0
    }

    fn abort_if_errors(&mut self) {
        self.emit_stashed_diagnostics();

        if self.has_errors() {
            FatalError.raise();
        }
    }

    fn span_bug(&mut self, sp: impl Into<MultiSpan>, msg: &str) -> ! {
        self.emit_diag_at_span(Diagnostic::new(Bug, msg), sp);
        panic::panic_any(ExplicitBug);
    }

    fn emit_diag_at_span(&mut self, mut diag: Diagnostic, sp: impl Into<MultiSpan>) {
        self.emit_diagnostic(diag.set_span(sp));
    }

    #[track_caller]
    fn delay_span_bug(&mut self, sp: impl Into<MultiSpan>, msg: &str) {
        // This is technically `self.treat_err_as_bug()` but `delay_span_bug` is called before
        // incrementing `err_count` by one, so we need to +1 the comparing.
        // FIXME: Would be nice to increment err_count in a more coherent way.
        if self.flags.treat_err_as_bug.map_or(false, |c| self.err_count() + 1 >= c.get()) {
            // FIXME: don't abort here if report_delayed_bugs is off
            self.span_bug(sp, msg);
        }
        let mut diagnostic = Diagnostic::new(Level::Bug, msg);
        diagnostic.set_span(sp.into());
        diagnostic.note(&format!("delayed at {}", std::panic::Location::caller()));
        self.delay_as_bug(diagnostic)
    }

    fn delay_good_path_bug(&mut self, msg: &str) {
        let diagnostic = Diagnostic::new(Level::Bug, msg);
        if self.flags.report_delayed_bugs {
            self.emit_diagnostic(&diagnostic);
        }
        let backtrace = std::backtrace::Backtrace::force_capture();
        self.delayed_good_path_bugs.push(DelayedDiagnostic::with_backtrace(diagnostic, backtrace));
    }

    fn failure(&mut self, msg: &str) {
        self.emit_diagnostic(&Diagnostic::new(FailureNote, msg));
    }

    fn fatal(&mut self, msg: &str) -> FatalError {
        self.emit_error(Fatal, msg);
        FatalError
    }

    fn err(&mut self, msg: &str) {
        self.emit_error(Error { lint: false }, msg);
    }

    /// Emit an error; level should be `Error` or `Fatal`.
    fn emit_error(&mut self, level: Level, msg: &str) {
        if self.treat_err_as_bug() {
            self.bug(msg);
        }
        self.emit_diagnostic(&Diagnostic::new(level, msg));
    }

    fn bug(&mut self, msg: &str) -> ! {
        self.emit_diagnostic(&Diagnostic::new(Bug, msg));
        panic::panic_any(ExplicitBug);
    }

    fn delay_as_bug(&mut self, diagnostic: Diagnostic) {
        if self.quiet {
            return;
        }
        if self.flags.report_delayed_bugs {
            self.emit_diagnostic(&diagnostic);
        }
        self.delayed_span_bugs.push(diagnostic);
    }

    fn flush_delayed(&mut self, bugs: Vec<Diagnostic>, explanation: &str) {
        let has_bugs = !bugs.is_empty();
        for bug in bugs {
            self.emit_diagnostic(&bug);
        }
        if has_bugs {
            panic!("{}", explanation);
        }
    }

    fn bump_lint_err_count(&mut self) {
        self.lint_err_count += 1;
        self.panic_if_treat_err_as_bug();
    }

    fn bump_err_count(&mut self) {
        self.err_count += 1;
        self.panic_if_treat_err_as_bug();
    }

    fn bump_warn_count(&mut self) {
        self.warn_count += 1;
    }

    fn panic_if_treat_err_as_bug(&self) {
        if self.treat_err_as_bug() {
            match (
                self.err_count() + self.lint_err_count,
                self.flags.treat_err_as_bug.map(|c| c.get()).unwrap_or(0),
            ) {
                (1, 1) => panic!("aborting due to `-Z treat-err-as-bug=1`"),
                (0, _) | (1, _) => {}
                (count, as_bug) => panic!(
                    "aborting after {} errors due to `-Z treat-err-as-bug={}`",
                    count, as_bug,
                ),
            }
        }
    }
}

struct DelayedDiagnostic {
    inner: Diagnostic,
    note: Backtrace,
}

impl DelayedDiagnostic {
    fn with_backtrace(diagnostic: Diagnostic, backtrace: Backtrace) -> Self {
        DelayedDiagnostic { inner: diagnostic, note: backtrace }
    }

    fn decorate(mut self) -> Diagnostic {
        self.inner.note(&format!("delayed at {}", self.note));
        self.inner
    }
}

#[derive(Copy, PartialEq, Clone, Hash, Debug, Encodable, Decodable)]
pub enum Level {
    Bug,
    Fatal,
    Error {
        /// If this error comes from a lint, don't abort compilation even when abort_if_errors() is called.
        lint: bool,
    },
    Warning,
    Note,
    Help,
    Cancelled,
    FailureNote,
    Allow,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_str().fmt(f)
    }
}

impl Level {
    fn color(self) -> ColorSpec {
        let mut spec = ColorSpec::new();
        match self {
            Bug | Fatal | Error { .. } => {
                spec.set_fg(Some(Color::Red)).set_intense(true);
            }
            Warning => {
                spec.set_fg(Some(Color::Yellow)).set_intense(cfg!(windows));
            }
            Note => {
                spec.set_fg(Some(Color::Green)).set_intense(true);
            }
            Help => {
                spec.set_fg(Some(Color::Cyan)).set_intense(true);
            }
            FailureNote => {}
            Allow | Cancelled => unreachable!(),
        }
        spec
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Bug => "error: internal compiler error",
            Fatal | Error { .. } => "error",
            Warning => "warning",
            Note => "note",
            Help => "help",
            FailureNote => "failure-note",
            Cancelled => panic!("Shouldn't call on cancelled error"),
            Allow => panic!("Shouldn't call on allowed error"),
        }
    }

    pub fn is_failure_note(&self) -> bool {
        matches!(*self, FailureNote)
    }
}

pub fn add_elided_lifetime_in_path_suggestion(
    source_map: &SourceMap,
    db: &mut DiagnosticBuilder<'_>,
    n: usize,
    path_span: Span,
    incl_angl_brckt: bool,
    insertion_span: Span,
    anon_lts: String,
) {
    let (replace_span, suggestion) = if incl_angl_brckt {
        (insertion_span, anon_lts)
    } else {
        // When possible, prefer a suggestion that replaces the whole
        // `Path<T>` expression with `Path<'_, T>`, rather than inserting `'_, `
        // at a point (which makes for an ugly/confusing label)
        if let Ok(snippet) = source_map.span_to_snippet(path_span) {
            // But our spans can get out of whack due to macros; if the place we think
            // we want to insert `'_` isn't even within the path expression's span, we
            // should bail out of making any suggestion rather than panicking on a
            // subtract-with-overflow or string-slice-out-out-bounds (!)
            // FIXME: can we do better?
            if insertion_span.lo().0 < path_span.lo().0 {
                return;
            }
            let insertion_index = (insertion_span.lo().0 - path_span.lo().0) as usize;
            if insertion_index > snippet.len() {
                return;
            }
            let (before, after) = snippet.split_at(insertion_index);
            (path_span, format!("{}{}{}", before, anon_lts, after))
        } else {
            (insertion_span, anon_lts)
        }
    };
    db.span_suggestion(
        replace_span,
        &format!("indicate the anonymous lifetime{}", pluralize!(n)),
        suggestion,
        Applicability::MachineApplicable,
    );
}

// Useful type to use with `Result<>` indicate that an error has already
// been reported to the user, so no need to continue checking.
#[derive(Clone, Copy, Debug, Encodable, Decodable, Hash, PartialEq, Eq)]
pub struct ErrorReported;

rustc_data_structures::impl_stable_hash_via_hash!(ErrorReported);
