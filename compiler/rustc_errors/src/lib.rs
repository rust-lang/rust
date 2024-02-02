//! Diagnostics creation and emission for `rustc`.
//!
//! This module contains the code for creating and emitting diagnostics.

// tidy-alphabetical-start
#![allow(incomplete_features)]
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(array_windows)]
#![feature(associated_type_defaults)]
#![feature(box_into_inner)]
#![feature(box_patterns)]
#![feature(error_reporter)]
#![feature(extract_if)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
// tidy-alphabetical-end

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

extern crate self as rustc_errors;

pub use codes::*;
pub use diagnostic::{
    AddToDiagnostic, DecorateLint, Diagnostic, DiagnosticArg, DiagnosticArgName,
    DiagnosticArgValue, DiagnosticStyledString, IntoDiagnosticArg, StringPart, SubDiagnostic,
};
pub use diagnostic_builder::{
    BugAbort, DiagnosticBuilder, EmissionGuarantee, FatalAbort, IntoDiagnostic,
};
pub use diagnostic_impls::{
    DiagnosticArgFromDisplay, DiagnosticSymbolList, ExpectedLifetimeParameter,
    IndicateAnonymousLifetime, InvalidFlushedDelayedDiagnosticLevel, SingleLabelManySpans,
};
pub use emitter::ColorConfig;
pub use rustc_error_messages::{
    fallback_fluent_bundle, fluent_bundle, DelayDm, DiagnosticMessage, FluentBundle,
    LanguageIdentifier, LazyFallbackBundle, MultiSpan, SpanLabel, SubdiagnosticMessage,
};
pub use rustc_lint_defs::{pluralize, Applicability};
pub use rustc_span::fatal_error::{FatalError, FatalErrorMarker};
pub use rustc_span::ErrorGuaranteed;
pub use snippet::Style;

// Used by external projects such as `rust-gpu`.
// See https://github.com/rust-lang/rust/pull/115393.
pub use termcolor::{Color, ColorSpec, WriteColor};

use crate::diagnostic_impls::{DelayedAtWithNewline, DelayedAtWithoutNewline};
use emitter::{is_case_difference, DynEmitter, Emitter, HumanEmitter};
use registry::Registry;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stable_hasher::{Hash128, StableHasher};
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_data_structures::AtomicRef;
use rustc_lint_defs::LintExpectationId;
use rustc_span::source_map::SourceMap;
use rustc_span::{Loc, Span, DUMMY_SP};
use std::backtrace::{Backtrace, BacktraceStatus};
use std::borrow::Cow;
use std::error::Report;
use std::fmt;
use std::hash::Hash;
use std::io::Write;
use std::num::NonZeroUsize;
use std::panic;
use std::path::{Path, PathBuf};

use Level::*;

pub mod annotate_snippet_emitter_writer;
pub mod codes;
mod diagnostic;
mod diagnostic_builder;
mod diagnostic_impls;
pub mod emitter;
pub mod error;
pub mod json;
mod lock;
pub mod markdown;
pub mod registry;
mod snippet;
mod styled_buffer;
#[cfg(test)]
mod tests;
pub mod translation;

pub type PErr<'a> = DiagnosticBuilder<'a>;
pub type PResult<'a, T> = Result<T, PErr<'a>>;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

// `PResult` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(PResult<'_, ()>, 16);
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

#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub struct CodeSuggestion {
    /// Each substitute can have multiple variants due to multiple
    /// applicable suggestions
    ///
    /// `foo.bar` might be replaced with `a.b` or `x.y` by replacing
    /// `foo` and `bar` on their own:
    ///
    /// ```ignore (illustrative)
    /// vec![
    ///     Substitution { parts: vec![(0..3, "a"), (4..7, "b")] },
    ///     Substitution { parts: vec![(0..3, "x"), (4..7, "y")] },
    /// ]
    /// ```
    ///
    /// or by replacing the entire span:
    ///
    /// ```ignore (illustrative)
    /// vec![
    ///     Substitution { parts: vec![(0..7, "a.b")] },
    ///     Substitution { parts: vec![(0..7, "x.y")] },
    /// ]
    /// ```
    pub substitutions: Vec<Substitution>,
    pub msg: DiagnosticMessage,
    /// Visual representation of this suggestion.
    pub style: SuggestionStyle,
    /// Whether or not the suggestion is approximate
    ///
    /// Sometimes we may show suggestions with placeholders,
    /// which are useful for users but not useful for
    /// tools like rustfix
    pub applicability: Applicability,
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
pub(crate) struct SubstitutionHighlight {
    start: usize,
    end: usize,
}

impl SubstitutionPart {
    pub fn is_addition(&self, sm: &SourceMap) -> bool {
        !self.snippet.is_empty() && !self.replaces_meaningful_content(sm)
    }

    pub fn is_deletion(&self, sm: &SourceMap) -> bool {
        self.snippet.trim().is_empty() && self.replaces_meaningful_content(sm)
    }

    pub fn is_replacement(&self, sm: &SourceMap) -> bool {
        !self.snippet.is_empty() && self.replaces_meaningful_content(sm)
    }

    fn replaces_meaningful_content(&self, sm: &SourceMap) -> bool {
        sm.span_to_snippet(self.span)
            .map_or(!self.span.is_empty(), |snippet| !snippet.trim().is_empty())
    }
}

impl CodeSuggestion {
    /// Returns the assembled code suggestions, whether they should be shown with an underline
    /// and whether the substitution only differs in capitalization.
    pub(crate) fn splice_lines(
        &self,
        sm: &SourceMap,
    ) -> Vec<(String, Vec<SubstitutionPart>, Vec<Vec<SubstitutionHighlight>>, bool)> {
        // For the `Vec<Vec<SubstitutionHighlight>>` value, the first level of the vector
        // corresponds to the output snippet's lines, while the second level corresponds to the
        // substrings within that line that should be highlighted.

        use rustc_span::{CharPos, Pos};

        /// Extracts a substring from the provided `line_opt` based on the specified low and high indices,
        /// appends it to the given buffer `buf`, and returns the count of newline characters in the substring
        /// for accurate highlighting.
        /// If `line_opt` is `None`, a newline character is appended to the buffer, and 0 is returned.
        ///
        /// ## Returns
        ///
        /// The count of newline characters in the extracted substring.
        fn push_trailing(
            buf: &mut String,
            line_opt: Option<&Cow<'_, str>>,
            lo: &Loc,
            hi_opt: Option<&Loc>,
        ) -> usize {
            let mut line_count = 0;
            // Convert CharPos to Usize, as CharPose is character offset
            // Extract low index and high index
            let (lo, hi_opt) = (lo.col.to_usize(), hi_opt.map(|hi| hi.col.to_usize()));
            if let Some(line) = line_opt {
                if let Some(lo) = line.char_indices().map(|(i, _)| i).nth(lo) {
                    // Get high index while account for rare unicode and emoji with char_indices
                    let hi_opt = hi_opt.and_then(|hi| line.char_indices().map(|(i, _)| i).nth(hi));
                    match hi_opt {
                        // If high index exist, take string from low to high index
                        Some(hi) if hi > lo => {
                            // count how many '\n' exist
                            line_count = line[lo..hi].matches('\n').count();
                            buf.push_str(&line[lo..hi])
                        }
                        Some(_) => (),
                        // If high index absence, take string from low index till end string.len
                        None => {
                            // count how many '\n' exist
                            line_count = line[lo..].matches('\n').count();
                            buf.push_str(&line[lo..])
                        }
                    }
                }
                // If high index is None
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
                if !sm.ensure_source_file_source_present(&lines.file) {
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
                    // Account for the difference between the width of the current code and the
                    // snippet being suggested, so that the *later* suggestions are correctly
                    // aligned on the screen. Note that cur_hi and cur_lo can be on different
                    // lines, so cur_hi.col can be smaller than cur_lo.col
                    acc += len - (cur_hi.col.0 as isize - cur_lo.col.0 as isize);
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

/// Signifies that the compiler died with an explicit call to `.bug`
/// or `.span_bug` rather than a failed assertion, etc.
pub struct ExplicitBug;

/// Signifies that the compiler died with an explicit call to `.delay_*_bug`
/// rather than a failed assertion, etc.
pub struct DelayedBugPanic;

/// A `DiagCtxt` deals with errors and other compiler output.
/// Certain errors (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct DiagCtxt {
    inner: Lock<DiagCtxtInner>,
}

/// This inner struct exists to keep it all behind a single lock;
/// this is done to prevent possible deadlocks in a multi-threaded compiler,
/// as well as inconsistent state observation.
struct DiagCtxtInner {
    flags: DiagCtxtFlags,

    /// The number of lint errors that have been emitted, including duplicates.
    lint_err_count: usize,
    /// The number of non-lint errors that have been emitted, including duplicates.
    err_count: usize,

    /// The error count shown to the user at the end.
    deduplicated_err_count: usize,
    /// The warning count shown to the user at the end.
    deduplicated_warn_count: usize,

    /// Has this diagnostic context printed any diagnostics? (I.e. has
    /// `self.emitter.emit_diagnostic()` been called?
    has_printed: bool,

    emitter: Box<DynEmitter>,
    span_delayed_bugs: Vec<DelayedDiagnostic>,
    good_path_delayed_bugs: Vec<DelayedDiagnostic>,
    /// This flag indicates that an expected diagnostic was emitted and suppressed.
    /// This is used for the `good_path_delayed_bugs` check.
    suppressed_expected_diag: bool,

    /// This set contains the code of all emitted diagnostics to avoid
    /// emitting the same diagnostic with extended help (`--teach`) twice, which
    /// would be unnecessary repetition.
    taught_diagnostics: FxHashSet<ErrCode>,

    /// Used to suggest rustc --explain `<error code>`
    emitted_diagnostic_codes: FxIndexSet<ErrCode>,

    /// This set contains a hash of every diagnostic that has been emitted by
    /// this `DiagCtxt`. These hashes is used to avoid emitting the same error
    /// twice.
    emitted_diagnostics: FxHashSet<Hash128>,

    /// Stashed diagnostics emitted in one stage of the compiler that may be
    /// stolen by other stages (e.g. to improve them and add more information).
    /// The stashed diagnostics count towards the total error count.
    /// When `.abort_if_errors()` is called, these are also emitted.
    stashed_diagnostics: FxIndexMap<(Span, StashKey), Diagnostic>,

    future_breakage_diagnostics: Vec<Diagnostic>,

    /// The [`Self::unstable_expect_diagnostics`] should be empty when this struct is
    /// dropped. However, it can have values if the compilation is stopped early
    /// or is only partially executed. To avoid ICEs, like in rust#94953 we only
    /// check if [`Self::unstable_expect_diagnostics`] is empty, if the expectation ids
    /// have been converted.
    check_unstable_expect_diagnostics: bool,

    /// Expected [`Diagnostic`][struct@diagnostic::Diagnostic]s store a [`LintExpectationId`] as part of
    /// the lint level. [`LintExpectationId`]s created early during the compilation
    /// (before `HirId`s have been defined) are not stable and can therefore not be
    /// stored on disk. This buffer stores these diagnostics until the ID has been
    /// replaced by a stable [`LintExpectationId`]. The [`Diagnostic`][struct@diagnostic::Diagnostic]s are the
    /// submitted for storage and added to the list of fulfilled expectations.
    unstable_expect_diagnostics: Vec<Diagnostic>,

    /// expected diagnostic will have the level `Expect` which additionally
    /// carries the [`LintExpectationId`] of the expectation that can be
    /// marked as fulfilled. This is a collection of all [`LintExpectationId`]s
    /// that have been marked as fulfilled this way.
    ///
    /// [RFC-2383]: https://rust-lang.github.io/rfcs/2383-lint-reasons.html
    fulfilled_expectations: FxHashSet<LintExpectationId>,

    /// The file where the ICE information is stored. This allows delayed_span_bug backtraces to be
    /// stored along side the main panic backtrace.
    ice_file: Option<PathBuf>,
}

/// A key denoting where from a diagnostic was stashed.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum StashKey {
    ItemNoType,
    UnderscoreForArrayLengths,
    EarlySyntaxWarning,
    CallIntoMethod,
    /// When an invalid lifetime e.g. `'2` should be reinterpreted
    /// as a char literal in the parser
    LifetimeIsChar,
    /// Maybe there was a typo where a comma was forgotten before
    /// FRU syntax
    MaybeFruTypo,
    CallAssocMethod,
    TraitMissingMethod,
    OpaqueHiddenTypeMismatch,
    MaybeForgetReturn,
    /// Query cycle detected, stashing in favor of a better error.
    Cycle,
    UndeterminedMacroResolution,
}

fn default_track_diagnostic(diag: Diagnostic, f: &mut dyn FnMut(Diagnostic)) {
    (*f)(diag)
}

pub static TRACK_DIAGNOSTIC: AtomicRef<fn(Diagnostic, &mut dyn FnMut(Diagnostic))> =
    AtomicRef::new(&(default_track_diagnostic as _));

#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug, Encodable, Decodable)]
pub enum DelayedBugKind {
    Normal,
    GoodPath,
}

#[derive(Copy, Clone, Default)]
pub struct DiagCtxtFlags {
    /// If false, warning-level lints are suppressed.
    /// (rustc: see `--allow warnings` and `--cap-lints`)
    pub can_emit_warnings: bool,
    /// If Some, the Nth error-level diagnostic is upgraded to bug-level.
    /// (rustc: see `-Z treat-err-as-bug`)
    pub treat_err_as_bug: Option<NonZeroUsize>,
    /// Eagerly emit delayed bugs as errors, so that the compiler debugger may
    /// see all of the errors being emitted at once.
    pub eagerly_emit_delayed_bugs: bool,
    /// Show macro backtraces.
    /// (rustc: see `-Z macro-backtrace`)
    pub macro_backtrace: bool,
    /// If true, identical diagnostics are reported only once.
    pub deduplicate_diagnostics: bool,
    /// Track where errors are created. Enabled with `-Ztrack-diagnostics`.
    pub track_diagnostics: bool,
}

impl Drop for DiagCtxtInner {
    fn drop(&mut self) {
        self.emit_stashed_diagnostics();

        if !self.has_errors() {
            self.flush_delayed(DelayedBugKind::Normal)
        }

        // FIXME(eddyb) this explains what `good_path_delayed_bugs` are!
        // They're `span_delayed_bugs` but for "require some diagnostic happened"
        // instead of "require some error happened". Sadly that isn't ideal, as
        // lints can be `#[allow]`'d, potentially leading to this triggering.
        // Also, "good path" should be replaced with a better naming.
        if !self.has_printed && !self.suppressed_expected_diag && !std::thread::panicking() {
            self.flush_delayed(DelayedBugKind::GoodPath);
        }

        if self.check_unstable_expect_diagnostics {
            assert!(
                self.unstable_expect_diagnostics.is_empty(),
                "all diagnostics with unstable expectations should have been converted",
            );
        }
    }
}

impl DiagCtxt {
    pub fn with_tty_emitter(
        sm: Option<Lrc<SourceMap>>,
        fallback_bundle: LazyFallbackBundle,
    ) -> Self {
        let emitter = Box::new(HumanEmitter::stderr(ColorConfig::Auto, fallback_bundle).sm(sm));
        Self::with_emitter(emitter)
    }
    pub fn disable_warnings(mut self) -> Self {
        self.inner.get_mut().flags.can_emit_warnings = false;
        self
    }

    pub fn with_flags(mut self, flags: DiagCtxtFlags) -> Self {
        self.inner.get_mut().flags = flags;
        self
    }

    pub fn with_ice_file(mut self, ice_file: PathBuf) -> Self {
        self.inner.get_mut().ice_file = Some(ice_file);
        self
    }

    pub fn with_emitter(emitter: Box<DynEmitter>) -> Self {
        Self {
            inner: Lock::new(DiagCtxtInner {
                flags: DiagCtxtFlags { can_emit_warnings: true, ..Default::default() },
                lint_err_count: 0,
                err_count: 0,
                deduplicated_err_count: 0,
                deduplicated_warn_count: 0,
                has_printed: false,
                emitter,
                span_delayed_bugs: Vec::new(),
                good_path_delayed_bugs: Vec::new(),
                suppressed_expected_diag: false,
                taught_diagnostics: Default::default(),
                emitted_diagnostic_codes: Default::default(),
                emitted_diagnostics: Default::default(),
                stashed_diagnostics: Default::default(),
                future_breakage_diagnostics: Vec::new(),
                check_unstable_expect_diagnostics: false,
                unstable_expect_diagnostics: Vec::new(),
                fulfilled_expectations: Default::default(),
                ice_file: None,
            }),
        }
    }

    /// Translate `message` eagerly with `args` to `SubdiagnosticMessage::Eager`.
    pub fn eagerly_translate<'a>(
        &self,
        message: DiagnosticMessage,
        args: impl Iterator<Item = DiagnosticArg<'a>>,
    ) -> SubdiagnosticMessage {
        SubdiagnosticMessage::Eager(Cow::from(self.eagerly_translate_to_string(message, args)))
    }

    /// Translate `message` eagerly with `args` to `String`.
    pub fn eagerly_translate_to_string<'a>(
        &self,
        message: DiagnosticMessage,
        args: impl Iterator<Item = DiagnosticArg<'a>>,
    ) -> String {
        let inner = self.inner.borrow();
        let args = crate::translation::to_fluent_args(args);
        inner.emitter.translate_message(&message, &args).map_err(Report::new).unwrap().to_string()
    }

    // This is here to not allow mutation of flags;
    // as of this writing it's only used in tests in librustc_middle.
    pub fn can_emit_warnings(&self) -> bool {
        self.inner.borrow_mut().flags.can_emit_warnings
    }

    /// Resets the diagnostic error count as well as the cached emitted diagnostics.
    ///
    /// NOTE: *do not* call this function from rustc. It is only meant to be called from external
    /// tools that want to reuse a `Parser` cleaning the previously emitted diagnostics as well as
    /// the overall count of emitted error diagnostics.
    pub fn reset_err_count(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.lint_err_count = 0;
        inner.err_count = 0;
        inner.deduplicated_err_count = 0;
        inner.deduplicated_warn_count = 0;
        inner.has_printed = false;

        // actually free the underlying memory (which `clear` would not do)
        inner.span_delayed_bugs = Default::default();
        inner.good_path_delayed_bugs = Default::default();
        inner.taught_diagnostics = Default::default();
        inner.emitted_diagnostic_codes = Default::default();
        inner.emitted_diagnostics = Default::default();
        inner.stashed_diagnostics = Default::default();
    }

    /// Stash a given diagnostic with the given `Span` and [`StashKey`] as the key.
    /// Retrieve a stashed diagnostic with `steal_diagnostic`.
    pub fn stash_diagnostic(&self, span: Span, key: StashKey, diag: Diagnostic) {
        let mut inner = self.inner.borrow_mut();

        let key = (span.with_parent(None), key);

        if diag.is_error() {
            if diag.is_lint.is_some() {
                inner.lint_err_count += 1;
            } else {
                inner.err_count += 1;
            }
        }

        // FIXME(Centril, #69537): Consider reintroducing panic on overwriting a stashed diagnostic
        // if/when we have a more robust macro-friendly replacement for `(span, key)` as a key.
        // See the PR for a discussion.
        inner.stashed_diagnostics.insert(key, diag);
    }

    /// Steal a previously stashed diagnostic with the given `Span` and [`StashKey`] as the key.
    pub fn steal_diagnostic(&self, span: Span, key: StashKey) -> Option<DiagnosticBuilder<'_, ()>> {
        let mut inner = self.inner.borrow_mut();
        let key = (span.with_parent(None), key);
        let diag = inner.stashed_diagnostics.remove(&key)?;
        if diag.is_error() {
            if diag.is_lint.is_some() {
                inner.lint_err_count -= 1;
            } else {
                inner.err_count -= 1;
            }
        }
        Some(DiagnosticBuilder::new_diagnostic(self, diag))
    }

    pub fn has_stashed_diagnostic(&self, span: Span, key: StashKey) -> bool {
        self.inner.borrow().stashed_diagnostics.get(&(span.with_parent(None), key)).is_some()
    }

    /// Emit all stashed diagnostics.
    pub fn emit_stashed_diagnostics(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow_mut().emit_stashed_diagnostics()
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    ///
    /// An `emit` call on the builder will only emit if `can_emit_warnings` is `true`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        self.struct_warn(msg).with_span(span)
    }

    /// Construct a builder at the `Warning` level with the `msg`.
    ///
    /// An `emit` call on the builder will only emit if `can_emit_warnings` is `true`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Warning, msg)
    }

    /// Construct a builder at the `Allow` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_allow(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Allow, msg)
    }

    /// Construct a builder at the `Expect` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_expect(
        &self,
        msg: impl Into<DiagnosticMessage>,
        id: LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Expect(id), msg)
    }

    /// Construct a builder at the `Error` level at the given `span` and with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_err(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_> {
        self.struct_err(msg).with_span(span)
    }

    /// Construct a builder at the `Error` level with the `msg`.
    // FIXME: This method should be removed (every error should have an associated error code).
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_err(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Error, msg)
    }

    /// Construct a builder at the `Fatal` level at the given `span` and with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_fatal(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, FatalAbort> {
        self.struct_fatal(msg).with_span(span)
    }

    /// Construct a builder at the `Fatal` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_fatal(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, FatalAbort> {
        DiagnosticBuilder::new(self, Fatal, msg)
    }

    /// Construct a builder at the `Help` level with the `msg`.
    #[rustc_lint_diagnostics]
    pub fn struct_help(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Help, msg)
    }

    /// Construct a builder at the `Note` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_note(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Note, msg)
    }

    /// Construct a builder at the `Bug` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_bug(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, BugAbort> {
        DiagnosticBuilder::new(self, Bug, msg)
    }

    /// Construct a builder at the `Bug` level at the given `span` with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_bug(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, BugAbort> {
        self.struct_bug(msg).with_span(span)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_fatal(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) -> ! {
        self.struct_span_fatal(span, msg).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_err(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        self.struct_span_err(span, msg).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_warn(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) {
        self.struct_span_warn(span, msg).emit()
    }

    pub fn span_bug(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) -> ! {
        self.struct_span_bug(span, msg).emit()
    }

    /// Ensures that compilation cannot succeed.
    ///
    /// If this function has been called but no errors have been emitted and
    /// compilation succeeds, it will cause an internal compiler error (ICE).
    ///
    /// This can be used in code paths that should never run on successful compilations.
    /// For example, it can be used to create an [`ErrorGuaranteed`]
    /// (but you should prefer threading through the [`ErrorGuaranteed`] from an error emission
    /// directly).
    #[track_caller]
    pub fn delayed_bug(&self, msg: impl Into<DiagnosticMessage>) -> ErrorGuaranteed {
        DiagnosticBuilder::<ErrorGuaranteed>::new(self, DelayedBug(DelayedBugKind::Normal), msg)
            .emit()
    }

    /// Like `delayed_bug`, but takes an additional span.
    ///
    /// Note: this function used to be called `delay_span_bug`. It was renamed
    /// to match similar functions like `span_err`, `span_warn`, etc.
    #[track_caller]
    pub fn span_delayed_bug(
        &self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        DiagnosticBuilder::<ErrorGuaranteed>::new(self, DelayedBug(DelayedBugKind::Normal), msg)
            .with_span(sp)
            .emit()
    }

    // FIXME(eddyb) note the comment inside `impl Drop for DiagCtxtInner`, that's
    // where the explanation of what "good path" is (also, it should be renamed).
    pub fn good_path_delayed_bug(&self, msg: impl Into<DiagnosticMessage>) {
        DiagnosticBuilder::<()>::new(self, DelayedBug(DelayedBugKind::GoodPath), msg).emit()
    }

    #[track_caller]
    #[rustc_lint_diagnostics]
    pub fn span_note(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) {
        self.struct_span_note(span, msg).emit()
    }

    #[track_caller]
    #[rustc_lint_diagnostics]
    pub fn struct_span_note(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Note, msg).with_span(span)
    }

    #[rustc_lint_diagnostics]
    pub fn fatal(&self, msg: impl Into<DiagnosticMessage>) -> ! {
        self.struct_fatal(msg).emit()
    }

    #[rustc_lint_diagnostics]
    pub fn err(&self, msg: impl Into<DiagnosticMessage>) -> ErrorGuaranteed {
        self.struct_err(msg).emit()
    }

    #[rustc_lint_diagnostics]
    pub fn warn(&self, msg: impl Into<DiagnosticMessage>) {
        self.struct_warn(msg).emit()
    }

    #[rustc_lint_diagnostics]
    pub fn note(&self, msg: impl Into<DiagnosticMessage>) {
        self.struct_note(msg).emit()
    }

    #[rustc_lint_diagnostics]
    pub fn bug(&self, msg: impl Into<DiagnosticMessage>) -> ! {
        self.struct_bug(msg).emit()
    }

    /// This excludes lint errors and delayed bugs.
    #[inline]
    pub fn err_count(&self) -> usize {
        self.inner.borrow().err_count
    }

    /// This excludes lint errors and delayed bugs.
    pub fn has_errors(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow().has_errors().then(|| {
            #[allow(deprecated)]
            ErrorGuaranteed::unchecked_claim_error_was_emitted()
        })
    }

    /// This excludes delayed bugs. Unless absolutely necessary, prefer
    /// `has_errors` to this method.
    pub fn has_errors_or_lint_errors(&self) -> Option<ErrorGuaranteed> {
        let inner = self.inner.borrow();
        let result = inner.has_errors() || inner.lint_err_count > 0;
        result.then(|| {
            #[allow(deprecated)]
            ErrorGuaranteed::unchecked_claim_error_was_emitted()
        })
    }

    /// Unless absolutely necessary, prefer `has_errors` or
    /// `has_errors_or_lint_errors` to this method.
    pub fn has_errors_or_lint_errors_or_delayed_bugs(&self) -> Option<ErrorGuaranteed> {
        let inner = self.inner.borrow();
        let result =
            inner.has_errors() || inner.lint_err_count > 0 || !inner.span_delayed_bugs.is_empty();
        result.then(|| {
            #[allow(deprecated)]
            ErrorGuaranteed::unchecked_claim_error_was_emitted()
        })
    }

    pub fn print_error_count(&self, registry: &Registry) {
        let mut inner = self.inner.borrow_mut();

        inner.emit_stashed_diagnostics();

        if inner.treat_err_as_bug() {
            return;
        }

        let warnings = match inner.deduplicated_warn_count {
            0 => Cow::from(""),
            1 => Cow::from("1 warning emitted"),
            count => Cow::from(format!("{count} warnings emitted")),
        };
        let errors = match inner.deduplicated_err_count {
            0 => Cow::from(""),
            1 => Cow::from("aborting due to 1 previous error"),
            count => Cow::from(format!("aborting due to {count} previous errors")),
        };

        match (errors.len(), warnings.len()) {
            (0, 0) => return,
            (0, _) => inner
                .emitter
                .emit_diagnostic(&Diagnostic::new(Warning, DiagnosticMessage::Str(warnings))),
            (_, 0) => {
                inner.emit_diagnostic(Diagnostic::new(Fatal, errors));
            }
            (_, _) => {
                inner.emit_diagnostic(Diagnostic::new(Fatal, format!("{errors}; {warnings}")));
            }
        }

        let can_show_explain = inner.emitter.should_show_explain();
        let are_there_diagnostics = !inner.emitted_diagnostic_codes.is_empty();
        if can_show_explain && are_there_diagnostics {
            let mut error_codes = inner
                .emitted_diagnostic_codes
                .iter()
                .filter_map(|&code| {
                    if registry.try_find_description(code).is_ok().clone() {
                        Some(code.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if !error_codes.is_empty() {
                error_codes.sort();
                if error_codes.len() > 1 {
                    let limit = if error_codes.len() > 9 { 9 } else { error_codes.len() };
                    inner.failure_note(format!(
                        "Some errors have detailed explanations: {}{}",
                        error_codes[..limit].join(", "),
                        if error_codes.len() > 9 { "..." } else { "." }
                    ));
                    inner.failure_note(format!(
                        "For more information about an error, try `rustc --explain {}`.",
                        &error_codes[0]
                    ));
                } else {
                    inner.failure_note(format!(
                        "For more information about this error, try `rustc --explain {}`.",
                        &error_codes[0]
                    ));
                }
            }
        }
    }

    pub fn abort_if_errors(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.emit_stashed_diagnostics();
        if inner.has_errors() {
            FatalError.raise();
        }
    }

    /// `true` if we haven't taught a diagnostic with this code already.
    /// The caller must then teach the user about such a diagnostic.
    ///
    /// Used to suppress emitting the same error multiple times with extended explanation when
    /// calling `-Zteach`.
    pub fn must_teach(&self, code: ErrCode) -> bool {
        self.inner.borrow_mut().taught_diagnostics.insert(code)
    }

    pub fn force_print_diagnostic(&self, db: Diagnostic) {
        self.inner.borrow_mut().emitter.emit_diagnostic(&db);
    }

    pub fn emit_diagnostic(&self, diagnostic: Diagnostic) -> Option<ErrorGuaranteed> {
        self.inner.borrow_mut().emit_diagnostic(diagnostic)
    }

    #[track_caller]
    pub fn emit_err<'a>(&'a self, err: impl IntoDiagnostic<'a>) -> ErrorGuaranteed {
        self.create_err(err).emit()
    }

    #[track_caller]
    pub fn create_err<'a>(&'a self, err: impl IntoDiagnostic<'a>) -> DiagnosticBuilder<'a> {
        err.into_diagnostic(self, Error)
    }

    #[track_caller]
    pub fn create_warn<'a>(
        &'a self,
        warning: impl IntoDiagnostic<'a, ()>,
    ) -> DiagnosticBuilder<'a, ()> {
        warning.into_diagnostic(self, Warning)
    }

    #[track_caller]
    pub fn emit_warn<'a>(&'a self, warning: impl IntoDiagnostic<'a, ()>) {
        self.create_warn(warning).emit()
    }

    #[track_caller]
    pub fn create_almost_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, FatalError>,
    ) -> DiagnosticBuilder<'a, FatalError> {
        fatal.into_diagnostic(self, Fatal)
    }

    #[track_caller]
    pub fn emit_almost_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, FatalError>,
    ) -> FatalError {
        self.create_almost_fatal(fatal).emit()
    }

    #[track_caller]
    pub fn create_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, FatalAbort>,
    ) -> DiagnosticBuilder<'a, FatalAbort> {
        fatal.into_diagnostic(self, Fatal)
    }

    #[track_caller]
    pub fn emit_fatal<'a>(&'a self, fatal: impl IntoDiagnostic<'a, FatalAbort>) -> ! {
        self.create_fatal(fatal).emit()
    }

    #[track_caller]
    pub fn create_bug<'a>(
        &'a self,
        bug: impl IntoDiagnostic<'a, BugAbort>,
    ) -> DiagnosticBuilder<'a, BugAbort> {
        bug.into_diagnostic(self, Bug)
    }

    #[track_caller]
    pub fn emit_bug<'a>(&'a self, bug: impl IntoDiagnostic<'a, diagnostic_builder::BugAbort>) -> ! {
        self.create_bug(bug).emit()
    }

    #[track_caller]
    pub fn emit_note<'a>(&'a self, note: impl IntoDiagnostic<'a, ()>) {
        self.create_note(note).emit()
    }

    #[track_caller]
    pub fn create_note<'a>(
        &'a self,
        note: impl IntoDiagnostic<'a, ()>,
    ) -> DiagnosticBuilder<'a, ()> {
        note.into_diagnostic(self, Note)
    }

    pub fn emit_artifact_notification(&self, path: &Path, artifact_type: &str) {
        self.inner.borrow_mut().emitter.emit_artifact_notification(path, artifact_type);
    }

    pub fn emit_future_breakage_report(&self) {
        let mut inner = self.inner.borrow_mut();
        let diags = std::mem::take(&mut inner.future_breakage_diagnostics);
        if !diags.is_empty() {
            inner.emitter.emit_future_breakage_report(diags);
        }
    }

    pub fn emit_unused_externs(
        &self,
        lint_level: rustc_lint_defs::Level,
        loud: bool,
        unused_externs: &[&str],
    ) {
        let mut inner = self.inner.borrow_mut();

        if loud && lint_level.is_error() {
            inner.lint_err_count += 1;
            inner.panic_if_treat_err_as_bug();
        }

        inner.emitter.emit_unused_externs(lint_level, unused_externs)
    }

    pub fn update_unstable_expectation_id(
        &self,
        unstable_to_stable: &FxIndexMap<LintExpectationId, LintExpectationId>,
    ) {
        let mut inner = self.inner.borrow_mut();
        let diags = std::mem::take(&mut inner.unstable_expect_diagnostics);
        inner.check_unstable_expect_diagnostics = true;

        if !diags.is_empty() {
            inner.suppressed_expected_diag = true;
            for mut diag in diags.into_iter() {
                diag.update_unstable_expectation_id(unstable_to_stable);

                // Here the diagnostic is given back to `emit_diagnostic` where it was first
                // intercepted. Now it should be processed as usual, since the unstable expectation
                // id is now stable.
                inner.emit_diagnostic(diag);
            }
        }

        inner
            .stashed_diagnostics
            .values_mut()
            .for_each(|diag| diag.update_unstable_expectation_id(unstable_to_stable));
        inner
            .future_breakage_diagnostics
            .iter_mut()
            .for_each(|diag| diag.update_unstable_expectation_id(unstable_to_stable));
    }

    /// This methods steals all [`LintExpectationId`]s that are stored inside
    /// [`DiagCtxtInner`] and indicate that the linked expectation has been fulfilled.
    #[must_use]
    pub fn steal_fulfilled_expectation_ids(&self) -> FxHashSet<LintExpectationId> {
        assert!(
            self.inner.borrow().unstable_expect_diagnostics.is_empty(),
            "`DiagCtxtInner::unstable_expect_diagnostics` should be empty at this point",
        );
        std::mem::take(&mut self.inner.borrow_mut().fulfilled_expectations)
    }

    pub fn flush_delayed(&self) {
        self.inner.borrow_mut().flush_delayed(DelayedBugKind::Normal);
    }
}

// Note: we prefer implementing operations on `DiagCtxt`, rather than
// `DiagCtxtInner`, whenever possible. This minimizes functions where
// `DiagCtxt::foo()` just borrows `inner` and forwards a call to
// `DiagCtxtInner::foo`.
impl DiagCtxtInner {
    /// Emit all stashed diagnostics.
    fn emit_stashed_diagnostics(&mut self) -> Option<ErrorGuaranteed> {
        let has_errors = self.has_errors();
        let mut reported = None;
        for (_, diag) in std::mem::take(&mut self.stashed_diagnostics).into_iter() {
            // Decrement the count tracking the stash; emitting will increment it.
            if diag.is_error() {
                if diag.is_lint.is_some() {
                    self.lint_err_count -= 1;
                } else {
                    self.err_count -= 1;
                }
            } else {
                // Unless they're forced, don't flush stashed warnings when
                // there are errors, to avoid causing warning overload. The
                // stash would've been stolen already if it were important.
                if !diag.is_force_warn() && has_errors {
                    continue;
                }
            }
            let reported_this = self.emit_diagnostic(diag);
            reported = reported.or(reported_this);
        }
        reported
    }

    fn emit_diagnostic(&mut self, mut diagnostic: Diagnostic) -> Option<ErrorGuaranteed> {
        // The `LintExpectationId` can be stable or unstable depending on when it was created.
        // Diagnostics created before the definition of `HirId`s are unstable and can not yet
        // be stored. Instead, they are buffered until the `LintExpectationId` is replaced by
        // a stable one by the `LintLevelsBuilder`.
        if let Some(LintExpectationId::Unstable { .. }) = diagnostic.level.get_expectation_id() {
            self.unstable_expect_diagnostics.push(diagnostic);
            return None;
        }

        // FIXME(eddyb) this should check for `has_errors` and stop pushing
        // once *any* errors were emitted (and truncate `span_delayed_bugs`
        // when an error is first emitted, also), but maybe there's a case
        // in which that's not sound? otherwise this is really inefficient.
        match diagnostic.level {
            DelayedBug(_) if self.flags.eagerly_emit_delayed_bugs => {
                diagnostic.level = Error;
            }
            DelayedBug(DelayedBugKind::Normal) => {
                let backtrace = std::backtrace::Backtrace::capture();
                self.span_delayed_bugs
                    .push(DelayedDiagnostic::with_backtrace(diagnostic, backtrace));
                #[allow(deprecated)]
                return Some(ErrorGuaranteed::unchecked_claim_error_was_emitted());
            }
            DelayedBug(DelayedBugKind::GoodPath) => {
                let backtrace = std::backtrace::Backtrace::capture();
                self.good_path_delayed_bugs
                    .push(DelayedDiagnostic::with_backtrace(diagnostic, backtrace));
                return None;
            }
            _ => {}
        }

        // This must come after the possible promotion of `DelayedBug` to
        // `Error` above.
        if matches!(diagnostic.level, Error | Fatal) && self.treat_next_err_as_bug() {
            diagnostic.level = Bug;
        }

        if diagnostic.has_future_breakage() {
            // Future breakages aren't emitted if they're Level::Allow,
            // but they still need to be constructed and stashed below,
            // so they'll trigger the good-path bug check.
            self.suppressed_expected_diag = true;
            self.future_breakage_diagnostics.push(diagnostic.clone());
        }

        if let Some(expectation_id) = diagnostic.level.get_expectation_id() {
            self.suppressed_expected_diag = true;
            self.fulfilled_expectations.insert(expectation_id.normalize());
        }

        if diagnostic.level == Warning && !self.flags.can_emit_warnings {
            if diagnostic.has_future_breakage() {
                (*TRACK_DIAGNOSTIC)(diagnostic, &mut |_| {});
            }
            return None;
        }

        if matches!(diagnostic.level, Expect(_) | Allow) {
            (*TRACK_DIAGNOSTIC)(diagnostic, &mut |_| {});
            return None;
        }

        let mut guaranteed = None;
        (*TRACK_DIAGNOSTIC)(diagnostic, &mut |mut diagnostic| {
            if let Some(code) = diagnostic.code {
                self.emitted_diagnostic_codes.insert(code);
            }

            let already_emitted = {
                let mut hasher = StableHasher::new();
                diagnostic.hash(&mut hasher);
                let diagnostic_hash = hasher.finish();
                !self.emitted_diagnostics.insert(diagnostic_hash)
            };

            // Only emit the diagnostic if we've been asked to deduplicate or
            // haven't already emitted an equivalent diagnostic.
            if !(self.flags.deduplicate_diagnostics && already_emitted) {
                debug!(?diagnostic);
                debug!(?self.emitted_diagnostics);
                let already_emitted_sub = |sub: &mut SubDiagnostic| {
                    debug!(?sub);
                    if sub.level != OnceNote && sub.level != OnceHelp {
                        return false;
                    }
                    let mut hasher = StableHasher::new();
                    sub.hash(&mut hasher);
                    let diagnostic_hash = hasher.finish();
                    debug!(?diagnostic_hash);
                    !self.emitted_diagnostics.insert(diagnostic_hash)
                };

                diagnostic.children.extract_if(already_emitted_sub).for_each(|_| {});
                if already_emitted {
                    diagnostic.note(
                        "duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`",
                    );
                }

                self.emitter.emit_diagnostic(&diagnostic);
                if diagnostic.is_error() {
                    self.deduplicated_err_count += 1;
                } else if matches!(diagnostic.level, ForceWarning(_) | Warning) {
                    self.deduplicated_warn_count += 1;
                }
                self.has_printed = true;
            }
            if diagnostic.is_error() {
                if diagnostic.is_lint.is_some() {
                    self.lint_err_count += 1;
                } else {
                    self.err_count += 1;
                }
                self.panic_if_treat_err_as_bug();

                #[allow(deprecated)]
                {
                    guaranteed = Some(ErrorGuaranteed::unchecked_claim_error_was_emitted());
                }
            }
        });

        guaranteed
    }

    fn treat_err_as_bug(&self) -> bool {
        self.flags.treat_err_as_bug.is_some_and(|c| self.err_count + self.lint_err_count >= c.get())
    }

    // Use this one before incrementing `err_count`.
    fn treat_next_err_as_bug(&self) -> bool {
        self.flags
            .treat_err_as_bug
            .is_some_and(|c| self.err_count + self.lint_err_count + 1 >= c.get())
    }

    fn has_errors(&self) -> bool {
        self.err_count > 0
    }

    fn failure_note(&mut self, msg: impl Into<DiagnosticMessage>) {
        self.emit_diagnostic(Diagnostic::new(FailureNote, msg));
    }

    fn flush_delayed(&mut self, kind: DelayedBugKind) {
        let (bugs, note1) = match kind {
            DelayedBugKind::Normal => (
                std::mem::take(&mut self.span_delayed_bugs),
                "no errors encountered even though `span_delayed_bug` issued",
            ),
            DelayedBugKind::GoodPath => (
                std::mem::take(&mut self.good_path_delayed_bugs),
                "no warnings or errors encountered even though `good_path_delayed_bugs` issued",
            ),
        };
        let note2 = "those delayed bugs will now be shown as internal compiler errors";

        if bugs.is_empty() {
            return;
        }

        // If backtraces are enabled, also print the query stack
        let backtrace = std::env::var_os("RUST_BACKTRACE").map_or(true, |x| &x != "0");
        for (i, bug) in bugs.into_iter().enumerate() {
            if let Some(file) = self.ice_file.as_ref()
                && let Ok(mut out) = std::fs::File::options().create(true).append(true).open(file)
            {
                let _ = write!(
                    &mut out,
                    "delayed span bug: {}\n{}\n",
                    bug.inner
                        .messages
                        .iter()
                        .filter_map(|(msg, _)| msg.as_str())
                        .collect::<String>(),
                    &bug.note
                );
            }

            if i == 0 {
                // Put the overall explanation before the `DelayedBug`s, to
                // frame them better (e.g. separate warnings from them). Also,
                // make it a note so it doesn't count as an error, because that
                // could trigger `-Ztreat-err-as-bug`, which we don't want.
                self.emit_diagnostic(Diagnostic::new(Note, note1));
                self.emit_diagnostic(Diagnostic::new(Note, note2));
            }

            let mut bug =
                if backtrace || self.ice_file.is_none() { bug.decorate() } else { bug.inner };

            // "Undelay" the `DelayedBug`s (into plain `Bug`s).
            if !matches!(bug.level, DelayedBug(_)) {
                // NOTE(eddyb) not panicking here because we're already producing
                // an ICE, and the more information the merrier.
                bug.subdiagnostic(InvalidFlushedDelayedDiagnosticLevel {
                    span: bug.span.primary_span().unwrap(),
                    level: bug.level,
                });
            }
            bug.level = Bug;

            self.emit_diagnostic(bug);
        }

        // Panic with `DelayedBugPanic` to avoid "unexpected panic" messages.
        panic::panic_any(DelayedBugPanic);
    }

    fn panic_if_treat_err_as_bug(&self) {
        if self.treat_err_as_bug() {
            let n = self.flags.treat_err_as_bug.map(|c| c.get()).unwrap();
            assert_eq!(n, self.err_count + self.lint_err_count);
            if n == 1 {
                panic!("aborting due to `-Z treat-err-as-bug=1`");
            } else {
                panic!("aborting after {n} errors due to `-Z treat-err-as-bug={n}`");
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
        match self.note.status() {
            BacktraceStatus::Captured => {
                let inner = &self.inner;
                self.inner.subdiagnostic(DelayedAtWithNewline {
                    span: inner.span.primary_span().unwrap_or(DUMMY_SP),
                    emitted_at: inner.emitted_at.clone(),
                    note: self.note,
                });
            }
            // Avoid the needless newline when no backtrace has been captured,
            // the display impl should just be a single line.
            _ => {
                let inner = &self.inner;
                self.inner.subdiagnostic(DelayedAtWithoutNewline {
                    span: inner.span.primary_span().unwrap_or(DUMMY_SP),
                    emitted_at: inner.emitted_at.clone(),
                    note: self.note,
                });
            }
        }

        self.inner
    }
}

#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug, Encodable, Decodable)]
pub enum Level {
    /// For bugs in the compiler. Manifests as an ICE (internal compiler error) panic.
    ///
    /// Its `EmissionGuarantee` is `BugAbort`.
    Bug,

    /// This is a strange one: lets you register an error without emitting it. If compilation ends
    /// without any other errors occurring, this will be emitted as a bug. Otherwise, it will be
    /// silently dropped. I.e. "expect other errors are emitted" semantics. Useful on code paths
    /// that should only be reached when compiling erroneous code.
    ///
    /// Its `EmissionGuarantee` is `ErrorGuaranteed` for `Normal` delayed bugs, and `()` for
    /// `GoodPath` delayed bugs.
    DelayedBug(DelayedBugKind),

    /// An error that causes an immediate abort. Used for things like configuration errors,
    /// internal overflows, some file operation errors.
    ///
    /// Its `EmissionGuarantee` is `FatalAbort`, except in the non-aborting "almost fatal" case
    /// that is occasionally used, where it is `FatalError`.
    Fatal,

    /// An error in the code being compiled, which prevents compilation from finishing. This is the
    /// most common case.
    ///
    /// Its `EmissionGuarantee` is `ErrorGuaranteed`.
    Error,

    /// A `force-warn` lint warning about the code being compiled. Does not prevent compilation
    /// from finishing.
    ///
    /// The [`LintExpectationId`] is used for expected lint diagnostics. In all other cases this
    /// should be `None`.
    ForceWarning(Option<LintExpectationId>),

    /// A warning about the code being compiled. Does not prevent compilation from finishing.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Warning,

    /// A message giving additional context. Rare, because notes are more commonly attached to other
    /// diagnostics such as errors.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Note,

    /// A note that is only emitted once. Rare, mostly used in circumstances relating to lints.
    ///
    /// Its `EmissionGuarantee` is `()`.
    OnceNote,

    /// A message suggesting how to fix something. Rare, because help messages are more commonly
    /// attached to other diagnostics such as errors.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Help,

    /// A help that is only emitted once. Rare.
    ///
    /// Its `EmissionGuarantee` is `()`.
    OnceHelp,

    /// Similar to `Note`, but used in cases where compilation has failed. Rare.
    ///
    /// Its `EmissionGuarantee` is `()`.
    FailureNote,

    /// Only used for lints.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Allow,

    /// Only used for lints.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Expect(LintExpectationId),
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
            Bug | DelayedBug(_) | Fatal | Error => {
                spec.set_fg(Some(Color::Red)).set_intense(true);
            }
            ForceWarning(_) | Warning => {
                spec.set_fg(Some(Color::Yellow)).set_intense(cfg!(windows));
            }
            Note | OnceNote => {
                spec.set_fg(Some(Color::Green)).set_intense(true);
            }
            Help | OnceHelp => {
                spec.set_fg(Some(Color::Cyan)).set_intense(true);
            }
            FailureNote => {}
            Allow | Expect(_) => unreachable!(),
        }
        spec
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Bug | DelayedBug(_) => "error: internal compiler error",
            Fatal | Error => "error",
            ForceWarning(_) | Warning => "warning",
            Note | OnceNote => "note",
            Help | OnceHelp => "help",
            FailureNote => "failure-note",
            Allow | Expect(_) => unreachable!(),
        }
    }

    pub fn is_failure_note(&self) -> bool {
        matches!(*self, FailureNote)
    }

    pub fn get_expectation_id(&self) -> Option<LintExpectationId> {
        match self {
            Expect(id) | ForceWarning(Some(id)) => Some(*id),
            _ => None,
        }
    }
}

// FIXME(eddyb) this doesn't belong here AFAICT, should be moved to callsite.
pub fn add_elided_lifetime_in_path_suggestion(
    source_map: &SourceMap,
    diag: &mut Diagnostic,
    n: usize,
    path_span: Span,
    incl_angl_brckt: bool,
    insertion_span: Span,
) {
    diag.subdiagnostic(ExpectedLifetimeParameter { span: path_span, count: n });
    if !source_map.is_span_accessible(insertion_span) {
        // Do not try to suggest anything if generated by a proc-macro.
        return;
    }
    let anon_lts = vec!["'_"; n].join(", ");
    let suggestion =
        if incl_angl_brckt { format!("<{anon_lts}>") } else { format!("{anon_lts}, ") };

    diag.subdiagnostic(IndicateAnonymousLifetime {
        span: insertion_span.shrink_to_hi(),
        count: n,
        suggestion,
    });
}

pub fn report_ambiguity_error<'a, G: EmissionGuarantee>(
    db: &mut DiagnosticBuilder<'a, G>,
    ambiguity: rustc_lint_defs::AmbiguityErrorDiag,
) {
    db.span_label(ambiguity.label_span, ambiguity.label_msg);
    db.note(ambiguity.note_msg);
    db.span_note(ambiguity.b1_span, ambiguity.b1_note_msg);
    for help_msg in ambiguity.b1_help_msgs {
        db.help(help_msg);
    }
    db.span_note(ambiguity.b2_span, ambiguity.b2_note_msg);
    for help_msg in ambiguity.b2_help_msgs {
        db.help(help_msg);
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum TerminalUrl {
    No,
    Yes,
    Auto,
}
