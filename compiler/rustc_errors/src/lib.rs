//! Diagnostics creation and emission for `rustc`.
//!
//! This module contains the code for creating and emitting diagnostics.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(array_windows)]
#![feature(drain_filter)]
#![feature(if_let_guard)]
#![feature(is_terminal)]
#![feature(adt_const_params)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(result_option_inspect)]
#![feature(rustc_attrs)]
#![feature(yeet_expr)]
#![feature(try_blocks)]
#![feature(box_patterns)]
#![feature(error_reporter)]
#![allow(incomplete_features)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

pub use emitter::ColorConfig;

use rustc_lint_defs::LintExpectationId;
use Level::*;

use emitter::{is_case_difference, Emitter, EmitterWriter};
use registry::Registry;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{self, Lock, Lrc};
use rustc_data_structures::AtomicRef;
pub use rustc_error_messages::{
    fallback_fluent_bundle, fluent, fluent_bundle, DelayDm, DiagnosticMessage, FluentBundle,
    LanguageIdentifier, LazyFallbackBundle, MultiSpan, SpanLabel, SubdiagnosticMessage,
    DEFAULT_LOCALE_RESOURCES,
};
pub use rustc_lint_defs::{pluralize, Applicability};
use rustc_span::source_map::SourceMap;
use rustc_span::HashStableContext;
use rustc_span::{Loc, Span};

use std::borrow::Cow;
use std::error::Report;
use std::fmt;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::panic;
use std::path::Path;

use termcolor::{Color, ColorSpec};

pub mod annotate_snippet_emitter_writer;
mod diagnostic;
mod diagnostic_builder;
mod diagnostic_impls;
pub mod emitter;
pub mod error;
pub mod json;
mod lock;
pub mod registry;
mod snippet;
mod styled_buffer;
#[cfg(test)]
mod tests;
pub mod translation;

pub use diagnostic_builder::IntoDiagnostic;
pub use snippet::Style;

pub type PErr<'a> = DiagnosticBuilder<'a, ErrorGuaranteed>;
pub type PResult<'a, T> = Result<T, PErr<'a>>;

// `PResult` is used a lot. Make sure it doesn't unintentionally get bigger.
// (See also the comment on `DiagnosticBuilderInner`'s `diagnostic` field.)
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
pub struct SubstitutionHighlight {
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
                        acc += len - (cur_hi.col.0 - cur_lo.col.0) as isize;
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
pub struct ExplicitBug;

/// Signifies that the compiler died with an explicit call to `.delay_*_bug`
/// rather than a failed assertion, etc.
pub struct DelayedBugPanic;

pub use diagnostic::{
    AddToDiagnostic, DecorateLint, Diagnostic, DiagnosticArg, DiagnosticArgValue, DiagnosticId,
    DiagnosticStyledString, IntoDiagnosticArg, SubDiagnostic,
};
pub use diagnostic_builder::{DiagnosticBuilder, EmissionGuarantee, Noted};
pub use diagnostic_impls::{DiagnosticArgFromDisplay, DiagnosticSymbolList};
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
    delayed_span_bugs: Vec<DelayedDiagnostic>,
    delayed_good_path_bugs: Vec<DelayedDiagnostic>,
    /// This flag indicates that an expected diagnostic was emitted and suppressed.
    /// This is used for the `delayed_good_path_bugs` check.
    suppressed_expected_diag: bool,

    /// This set contains the `DiagnosticId` of all emitted diagnostics to avoid
    /// emitting the same diagnostic with extended help (`--teach`) twice, which
    /// would be unnecessary repetition.
    taught_diagnostics: FxHashSet<DiagnosticId>,

    /// Used to suggest rustc --explain `<error code>`
    emitted_diagnostic_codes: FxIndexSet<DiagnosticId>,

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

    /// The [`Self::unstable_expect_diagnostics`] should be empty when this struct is
    /// dropped. However, it can have values if the compilation is stopped early
    /// or is only partially executed. To avoid ICEs, like in rust#94953 we only
    /// check if [`Self::unstable_expect_diagnostics`] is empty, if the expectation ids
    /// have been converted.
    check_unstable_expect_diagnostics: bool,

    /// Expected [`Diagnostic`][diagnostic::Diagnostic]s store a [`LintExpectationId`] as part of
    /// the lint level. [`LintExpectationId`]s created early during the compilation
    /// (before `HirId`s have been defined) are not stable and can therefore not be
    /// stored on disk. This buffer stores these diagnostics until the ID has been
    /// replaced by a stable [`LintExpectationId`]. The [`Diagnostic`][diagnostic::Diagnostic]s are the
    /// submitted for storage and added to the list of fulfilled expectations.
    unstable_expect_diagnostics: Vec<Diagnostic>,

    /// expected diagnostic will have the level `Expect` which additionally
    /// carries the [`LintExpectationId`] of the expectation that can be
    /// marked as fulfilled. This is a collection of all [`LintExpectationId`]s
    /// that have been marked as fulfilled this way.
    ///
    /// [RFC-2383]: https://rust-lang.github.io/rfcs/2383-lint-reasons.html
    fulfilled_expectations: FxHashSet<LintExpectationId>,
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
}

fn default_track_diagnostic(d: &mut Diagnostic, f: &mut dyn FnMut(&mut Diagnostic)) {
    (*f)(d)
}

pub static TRACK_DIAGNOSTICS: AtomicRef<fn(&mut Diagnostic, &mut dyn FnMut(&mut Diagnostic))> =
    AtomicRef::new(&(default_track_diagnostic as _));

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
    /// Track where errors are created. Enabled with `-Ztrack-diagnostics`.
    pub track_diagnostics: bool,
}

impl Drop for HandlerInner {
    fn drop(&mut self) {
        self.emit_stashed_diagnostics();

        if !self.has_errors() {
            let bugs = std::mem::replace(&mut self.delayed_span_bugs, Vec::new());
            self.flush_delayed(bugs, "no errors encountered even though `delay_span_bug` issued");
        }

        // FIXME(eddyb) this explains what `delayed_good_path_bugs` are!
        // They're `delayed_span_bugs` but for "require some diagnostic happened"
        // instead of "require some error happened". Sadly that isn't ideal, as
        // lints can be `#[allow]`'d, potentially leading to this triggering.
        // Also, "good path" should be replaced with a better naming.
        if !self.has_any_message() && !self.suppressed_expected_diag {
            let bugs = std::mem::replace(&mut self.delayed_good_path_bugs, Vec::new());
            self.flush_delayed(
                bugs,
                "no warnings or errors encountered even though `delayed_good_path_bugs` issued",
            );
        }

        if self.check_unstable_expect_diagnostics {
            assert!(
                self.unstable_expect_diagnostics.is_empty(),
                "all diagnostics with unstable expectations should have been converted",
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
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
    ) -> Self {
        Self::with_tty_emitter_and_flags(
            color_config,
            sm,
            fluent_bundle,
            fallback_bundle,
            HandlerFlags { can_emit_warnings, treat_err_as_bug, ..Default::default() },
        )
    }

    pub fn with_tty_emitter_and_flags(
        color_config: ColorConfig,
        sm: Option<Lrc<SourceMap>>,
        fluent_bundle: Option<Lrc<FluentBundle>>,
        fallback_bundle: LazyFallbackBundle,
        flags: HandlerFlags,
    ) -> Self {
        let emitter = Box::new(EmitterWriter::stderr(
            color_config,
            sm,
            fluent_bundle,
            fallback_bundle,
            false,
            false,
            None,
            flags.macro_backtrace,
            flags.track_diagnostics,
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
                suppressed_expected_diag: false,
                taught_diagnostics: Default::default(),
                emitted_diagnostic_codes: Default::default(),
                emitted_diagnostics: Default::default(),
                stashed_diagnostics: Default::default(),
                future_breakage_diagnostics: Vec::new(),
                check_unstable_expect_diagnostics: false,
                unstable_expect_diagnostics: Vec::new(),
                fulfilled_expectations: Default::default(),
            }),
        }
    }

    /// Translate `message` eagerly with `args` to `SubdiagnosticMessage::Eager`.
    pub fn eagerly_translate<'a>(
        &self,
        message: DiagnosticMessage,
        args: impl Iterator<Item = DiagnosticArg<'a, 'static>>,
    ) -> SubdiagnosticMessage {
        SubdiagnosticMessage::Eager(self.eagerly_translate_to_string(message, args))
    }

    /// Translate `message` eagerly with `args` to `String`.
    pub fn eagerly_translate_to_string<'a>(
        &self,
        message: DiagnosticMessage,
        args: impl Iterator<Item = DiagnosticArg<'a, 'static>>,
    ) -> String {
        let inner = self.inner.borrow();
        let args = crate::translation::to_fluent_args(args);
        inner.emitter.translate_message(&message, &args).map_err(Report::new).unwrap().to_string()
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

    /// Stash a given diagnostic with the given `Span` and [`StashKey`] as the key.
    /// Retrieve a stashed diagnostic with `steal_diagnostic`.
    pub fn stash_diagnostic(&self, span: Span, key: StashKey, diag: Diagnostic) {
        let mut inner = self.inner.borrow_mut();
        inner.stash((span.with_parent(None), key), diag);
    }

    /// Steal a previously stashed diagnostic with the given `Span` and [`StashKey`] as the key.
    pub fn steal_diagnostic(&self, span: Span, key: StashKey) -> Option<DiagnosticBuilder<'_, ()>> {
        let mut inner = self.inner.borrow_mut();
        inner
            .steal((span.with_parent(None), key))
            .map(|diag| DiagnosticBuilder::new_diagnostic(self, diag))
    }

    pub fn has_stashed_diagnostic(&self, span: Span, key: StashKey) -> bool {
        self.inner.borrow().stashed_diagnostics.get(&(span.with_parent(None), key)).is_some()
    }

    /// Emit all stashed diagnostics.
    pub fn emit_stashed_diagnostics(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow_mut().emit_stashed_diagnostics()
    }

    /// Construct a builder with the `msg` at the level appropriate for the specific `EmissionGuarantee`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_diagnostic<G: EmissionGuarantee>(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, G> {
        G::make_diagnostic_builder(self, msg)
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    ///
    /// Attempting to `.emit()` the builder will only emit if either:
    /// * `can_emit_warnings` is `true`
    /// * `is_force_warn` was set in `DiagnosticId::Lint`
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        let mut result = self.struct_warn(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    /// The `id` is used for lint emissions which should also fulfill a lint expectation.
    ///
    /// Attempting to `.emit()` the builder will only emit if either:
    /// * `can_emit_warnings` is `true`
    /// * `is_force_warn` was set in `DiagnosticId::Lint`
    #[track_caller]
    pub fn struct_span_warn_with_expectation(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        id: LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        let mut result = self.struct_warn_with_expectation(msg, id);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Allow` level at the given `span` and with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_allow(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        let mut result = self.struct_allow(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Warning` level at the given `span` and with the `msg`.
    /// Also include a code.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ()> {
        let mut result = self.struct_span_warn(span, msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Warning` level with the `msg`.
    ///
    /// Attempting to `.emit()` the builder will only emit if either:
    /// * `can_emit_warnings` is `true`
    /// * `is_force_warn` was set in `DiagnosticId::Lint`
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Warning(None), msg)
    }

    /// Construct a builder at the `Warning` level with the `msg`. The `id` is used for
    /// lint emissions which should also fulfill a lint expectation.
    ///
    /// Attempting to `.emit()` the builder will only emit if either:
    /// * `can_emit_warnings` is `true`
    /// * `is_force_warn` was set in `DiagnosticId::Lint`
    #[track_caller]
    pub fn struct_warn_with_expectation(
        &self,
        msg: impl Into<DiagnosticMessage>,
        id: LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Warning(Some(id)), msg)
    }

    /// Construct a builder at the `Allow` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_allow(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Allow, msg)
    }

    /// Construct a builder at the `Expect` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_expect(
        &self,
        msg: impl Into<DiagnosticMessage>,
        id: LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Expect(id), msg)
    }

    /// Construct a builder at the `Error` level at the given `span` and with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_err(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut result = self.struct_err(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Error` level at the given `span`, with the `msg`, and `code`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_err_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut result = self.struct_span_err(span, msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Error` level with the `msg`.
    // FIXME: This method should be removed (every error should have an associated error code).
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_err(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        DiagnosticBuilder::new_guaranteeing_error::<_, { Level::Error { lint: false } }>(self, msg)
    }

    /// This should only be used by `rustc_middle::lint::struct_lint_level`. Do not use it for hard errors.
    #[doc(hidden)]
    #[track_caller]
    pub fn struct_err_lint(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Error { lint: true }, msg)
    }

    /// Construct a builder at the `Error` level with the `msg` and the `code`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_err_with_code(
        &self,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut result = self.struct_err(msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Warn` level with the `msg` and the `code`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn_with_code(
        &self,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ()> {
        let mut result = self.struct_warn(msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Fatal` level at the given `span` and with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_fatal(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, !> {
        let mut result = self.struct_fatal(msg);
        result.set_span(span);
        result
    }

    /// Construct a builder at the `Fatal` level at the given `span`, with the `msg`, and `code`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_fatal_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, !> {
        let mut result = self.struct_span_fatal(span, msg);
        result.code(code);
        result
    }

    /// Construct a builder at the `Error` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_fatal(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, !> {
        DiagnosticBuilder::new_fatal(self, msg)
    }

    /// Construct a builder at the `Help` level with the `msg`.
    #[rustc_lint_diagnostics]
    pub fn struct_help(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Help, msg)
    }

    /// Construct a builder at the `Note` level with the `msg`.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_note_without_error(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        DiagnosticBuilder::new(self, Level::Note, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_fatal(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) -> ! {
        self.emit_diag_at_span(Diagnostic::new(Fatal, msg), span);
        FatalError.raise()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_fatal_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> ! {
        self.emit_diag_at_span(Diagnostic::new_with_code(Fatal, Some(code), msg), span);
        FatalError.raise()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_err(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        self.emit_diag_at_span(Diagnostic::new(Error { lint: false }, msg), span).unwrap()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_err_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) {
        self.emit_diag_at_span(
            Diagnostic::new_with_code(Error { lint: false }, Some(code), msg),
            span,
        );
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_warn(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) {
        self.emit_diag_at_span(Diagnostic::new(Warning(None), msg), span);
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_warn_with_code(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) {
        self.emit_diag_at_span(Diagnostic::new_with_code(Warning(None), Some(code), msg), span);
    }

    pub fn span_bug(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) -> ! {
        self.inner.borrow_mut().span_bug(span, msg)
    }

    /// For documentation on this, see `Session::delay_span_bug`.
    #[track_caller]
    pub fn delay_span_bug(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        self.inner.borrow_mut().delay_span_bug(span, msg)
    }

    // FIXME(eddyb) note the comment inside `impl Drop for HandlerInner`, that's
    // where the explanation of what "good path" is (also, it should be renamed).
    pub fn delay_good_path_bug(&self, msg: impl Into<DiagnosticMessage>) {
        self.inner.borrow_mut().delay_good_path_bug(msg)
    }

    #[track_caller]
    pub fn span_bug_no_panic(&self, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) {
        self.emit_diag_at_span(Diagnostic::new(Bug, msg), span);
    }

    #[track_caller]
    #[rustc_lint_diagnostics]
    pub fn span_note_without_error(
        &self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) {
        self.emit_diag_at_span(Diagnostic::new(Note, msg), span);
    }

    #[track_caller]
    #[rustc_lint_diagnostics]
    pub fn span_note_diag(
        &self,
        span: Span,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        let mut db = DiagnosticBuilder::new(self, Note, msg);
        db.set_span(span);
        db
    }

    // NOTE: intentionally doesn't raise an error so rustc_codegen_ssa only reports fatal errors in the main thread
    #[rustc_lint_diagnostics]
    pub fn fatal(&self, msg: impl Into<DiagnosticMessage>) -> FatalError {
        self.inner.borrow_mut().fatal(msg)
    }

    #[rustc_lint_diagnostics]
    pub fn err(&self, msg: impl Into<DiagnosticMessage>) -> ErrorGuaranteed {
        self.inner.borrow_mut().err(msg)
    }

    #[rustc_lint_diagnostics]
    pub fn warn(&self, msg: impl Into<DiagnosticMessage>) {
        let mut db = DiagnosticBuilder::new(self, Warning(None), msg);
        db.emit();
    }

    #[rustc_lint_diagnostics]
    pub fn note_without_error(&self, msg: impl Into<DiagnosticMessage>) {
        DiagnosticBuilder::new(self, Note, msg).emit();
    }

    pub fn bug(&self, msg: impl Into<DiagnosticMessage>) -> ! {
        self.inner.borrow_mut().bug(msg)
    }

    #[inline]
    pub fn err_count(&self) -> usize {
        self.inner.borrow().err_count()
    }

    pub fn has_errors(&self) -> Option<ErrorGuaranteed> {
        if self.inner.borrow().has_errors() { Some(ErrorGuaranteed(())) } else { None }
    }

    pub fn has_errors_or_lint_errors(&self) -> Option<ErrorGuaranteed> {
        if self.inner.borrow().has_errors_or_lint_errors() {
            Some(ErrorGuaranteed::unchecked_claim_error_was_emitted())
        } else {
            None
        }
    }
    pub fn has_errors_or_delayed_span_bugs(&self) -> Option<ErrorGuaranteed> {
        if self.inner.borrow().has_errors_or_delayed_span_bugs() {
            Some(ErrorGuaranteed::unchecked_claim_error_was_emitted())
        } else {
            None
        }
    }
    pub fn is_compilation_going_to_fail(&self) -> Option<ErrorGuaranteed> {
        if self.inner.borrow().is_compilation_going_to_fail() {
            Some(ErrorGuaranteed::unchecked_claim_error_was_emitted())
        } else {
            None
        }
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

    pub fn emit_diagnostic(&self, diagnostic: &mut Diagnostic) -> Option<ErrorGuaranteed> {
        self.inner.borrow_mut().emit_diagnostic(diagnostic)
    }

    pub fn emit_err<'a>(&'a self, err: impl IntoDiagnostic<'a>) -> ErrorGuaranteed {
        self.create_err(err).emit()
    }

    pub fn create_err<'a>(
        &'a self,
        err: impl IntoDiagnostic<'a>,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        err.into_diagnostic(self)
    }

    pub fn create_warning<'a>(
        &'a self,
        warning: impl IntoDiagnostic<'a, ()>,
    ) -> DiagnosticBuilder<'a, ()> {
        warning.into_diagnostic(self)
    }

    pub fn emit_warning<'a>(&'a self, warning: impl IntoDiagnostic<'a, ()>) {
        self.create_warning(warning).emit()
    }

    pub fn create_almost_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, FatalError>,
    ) -> DiagnosticBuilder<'a, FatalError> {
        fatal.into_diagnostic(self)
    }

    pub fn emit_almost_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, FatalError>,
    ) -> FatalError {
        self.create_almost_fatal(fatal).emit()
    }

    pub fn create_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, !>,
    ) -> DiagnosticBuilder<'a, !> {
        fatal.into_diagnostic(self)
    }

    pub fn emit_fatal<'a>(&'a self, fatal: impl IntoDiagnostic<'a, !>) -> ! {
        self.create_fatal(fatal).emit()
    }

    pub fn create_bug<'a>(
        &'a self,
        bug: impl IntoDiagnostic<'a, diagnostic_builder::Bug>,
    ) -> DiagnosticBuilder<'a, diagnostic_builder::Bug> {
        bug.into_diagnostic(self)
    }

    pub fn emit_bug<'a>(
        &'a self,
        bug: impl IntoDiagnostic<'a, diagnostic_builder::Bug>,
    ) -> diagnostic_builder::Bug {
        self.create_bug(bug).emit()
    }

    pub fn emit_note<'a>(&'a self, note: impl IntoDiagnostic<'a, Noted>) -> Noted {
        self.create_note(note).emit()
    }

    pub fn create_note<'a>(
        &'a self,
        note: impl IntoDiagnostic<'a, Noted>,
    ) -> DiagnosticBuilder<'a, Noted> {
        note.into_diagnostic(self)
    }

    fn emit_diag_at_span(
        &self,
        mut diag: Diagnostic,
        sp: impl Into<MultiSpan>,
    ) -> Option<ErrorGuaranteed> {
        let mut inner = self.inner.borrow_mut();
        inner.emit_diagnostic(diag.set_span(sp))
    }

    pub fn emit_artifact_notification(&self, path: &Path, artifact_type: &str) {
        self.inner.borrow_mut().emit_artifact_notification(path, artifact_type)
    }

    pub fn emit_future_breakage_report(&self, diags: Vec<Diagnostic>) {
        self.inner.borrow_mut().emitter.emit_future_breakage_report(diags)
    }

    pub fn emit_unused_externs(
        &self,
        lint_level: rustc_lint_defs::Level,
        loud: bool,
        unused_externs: &[&str],
    ) {
        let mut inner = self.inner.borrow_mut();

        if loud && lint_level.is_error() {
            inner.bump_err_count();
        }

        inner.emit_unused_externs(lint_level, unused_externs)
    }

    pub fn update_unstable_expectation_id(
        &self,
        unstable_to_stable: &FxHashMap<LintExpectationId, LintExpectationId>,
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
                inner.emit_diagnostic(&mut diag);
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
    /// [`HandlerInner`] and indicate that the linked expectation has been fulfilled.
    #[must_use]
    pub fn steal_fulfilled_expectation_ids(&self) -> FxHashSet<LintExpectationId> {
        assert!(
            self.inner.borrow().unstable_expect_diagnostics.is_empty(),
            "`HandlerInner::unstable_expect_diagnostics` should be empty at this point",
        );
        std::mem::take(&mut self.inner.borrow_mut().fulfilled_expectations)
    }

    pub fn flush_delayed(&self) {
        let mut inner = self.inner.lock();
        let bugs = std::mem::replace(&mut inner.delayed_span_bugs, Vec::new());
        inner.flush_delayed(bugs, "no errors encountered even though `delay_span_bug` issued");
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
    fn emit_stashed_diagnostics(&mut self) -> Option<ErrorGuaranteed> {
        let has_errors = self.has_errors();
        let diags = self.stashed_diagnostics.drain(..).map(|x| x.1).collect::<Vec<_>>();
        let mut reported = None;
        for mut diag in diags {
            // Decrement the count tracking the stash; emitting will increment it.
            if diag.is_error() {
                if matches!(diag.level, Level::Error { lint: true }) {
                    self.lint_err_count -= 1;
                } else {
                    self.err_count -= 1;
                }
            } else {
                if diag.is_force_warn() {
                    self.warn_count -= 1;
                } else {
                    // Unless they're forced, don't flush stashed warnings when
                    // there are errors, to avoid causing warning overload. The
                    // stash would've been stolen already if it were important.
                    if has_errors {
                        continue;
                    }
                }
            }
            let reported_this = self.emit_diagnostic(&mut diag);
            reported = reported.or(reported_this);
        }
        reported
    }

    // FIXME(eddyb) this should ideally take `diagnostic` by value.
    fn emit_diagnostic(&mut self, diagnostic: &mut Diagnostic) -> Option<ErrorGuaranteed> {
        // The `LintExpectationId` can be stable or unstable depending on when it was created.
        // Diagnostics created before the definition of `HirId`s are unstable and can not yet
        // be stored. Instead, they are buffered until the `LintExpectationId` is replaced by
        // a stable one by the `LintLevelsBuilder`.
        if let Some(LintExpectationId::Unstable { .. }) = diagnostic.level.get_expectation_id() {
            self.unstable_expect_diagnostics.push(diagnostic.clone());
            return None;
        }

        if diagnostic.level == Level::DelayedBug {
            // FIXME(eddyb) this should check for `has_errors` and stop pushing
            // once *any* errors were emitted (and truncate `delayed_span_bugs`
            // when an error is first emitted, also), but maybe there's a case
            // in which that's not sound? otherwise this is really inefficient.
            let backtrace = std::backtrace::Backtrace::force_capture();
            self.delayed_span_bugs
                .push(DelayedDiagnostic::with_backtrace(diagnostic.clone(), backtrace));

            if !self.flags.report_delayed_bugs {
                return Some(ErrorGuaranteed::unchecked_claim_error_was_emitted());
            }
        }

        if diagnostic.has_future_breakage() {
            // Future breakages aren't emitted if they're Level::Allowed,
            // but they still need to be constructed and stashed below,
            // so they'll trigger the good-path bug check.
            self.suppressed_expected_diag = true;
            self.future_breakage_diagnostics.push(diagnostic.clone());
        }

        if let Some(expectation_id) = diagnostic.level.get_expectation_id() {
            self.suppressed_expected_diag = true;
            self.fulfilled_expectations.insert(expectation_id.normalize());
        }

        if matches!(diagnostic.level, Warning(_))
            && !self.flags.can_emit_warnings
            && !diagnostic.is_force_warn()
        {
            if diagnostic.has_future_breakage() {
                (*TRACK_DIAGNOSTICS)(diagnostic, &mut |_| {});
            }
            return None;
        }

        if matches!(diagnostic.level, Level::Expect(_) | Level::Allow) {
            (*TRACK_DIAGNOSTICS)(diagnostic, &mut |_| {});
            return None;
        }

        let mut guaranteed = None;
        (*TRACK_DIAGNOSTICS)(diagnostic, &mut |diagnostic| {
            if let Some(ref code) = diagnostic.code {
                self.emitted_diagnostic_codes.insert(code.clone());
            }

            let already_emitted = |this: &mut Self| {
                let mut hasher = StableHasher::new();
                diagnostic.hash(&mut hasher);
                let diagnostic_hash = hasher.finish();
                !this.emitted_diagnostics.insert(diagnostic_hash)
            };

            // Only emit the diagnostic if we've been asked to deduplicate or
            // haven't already emitted an equivalent diagnostic.
            if !(self.flags.deduplicate_diagnostics && already_emitted(self)) {
                debug!(?diagnostic);
                debug!(?self.emitted_diagnostics);
                let already_emitted_sub = |sub: &mut SubDiagnostic| {
                    debug!(?sub);
                    if sub.level != Level::OnceNote {
                        return false;
                    }
                    let mut hasher = StableHasher::new();
                    sub.hash(&mut hasher);
                    let diagnostic_hash = hasher.finish();
                    debug!(?diagnostic_hash);
                    !self.emitted_diagnostics.insert(diagnostic_hash)
                };

                diagnostic.children.drain_filter(already_emitted_sub).for_each(|_| {});

                self.emitter.emit_diagnostic(diagnostic);
                if diagnostic.is_error() {
                    self.deduplicated_err_count += 1;
                } else if let Warning(_) = diagnostic.level {
                    self.deduplicated_warn_count += 1;
                }
            }
            if diagnostic.is_error() {
                if matches!(diagnostic.level, Level::Error { lint: true }) {
                    self.bump_lint_err_count();
                } else {
                    self.bump_err_count();
                }

                guaranteed = Some(ErrorGuaranteed::unchecked_claim_error_was_emitted());
            } else {
                self.bump_warn_count();
            }
        });

        guaranteed
    }

    fn emit_artifact_notification(&mut self, path: &Path, artifact_type: &str) {
        self.emitter.emit_artifact_notification(path, artifact_type);
    }

    fn emit_unused_externs(&mut self, lint_level: rustc_lint_defs::Level, unused_externs: &[&str]) {
        self.emitter.emit_unused_externs(lint_level, unused_externs);
    }

    fn treat_err_as_bug(&self) -> bool {
        self.flags.treat_err_as_bug.map_or(false, |c| {
            self.err_count() + self.lint_err_count + self.delayed_bug_count() >= c.get()
        })
    }

    fn delayed_bug_count(&self) -> usize {
        self.delayed_span_bugs.len() + self.delayed_good_path_bugs.len()
    }

    fn print_error_count(&mut self, registry: &Registry) {
        self.emit_stashed_diagnostics();

        let warnings = match self.deduplicated_warn_count {
            0 => String::new(),
            1 => "1 warning emitted".to_string(),
            count => format!("{count} warnings emitted"),
        };
        let errors = match self.deduplicated_err_count {
            0 => String::new(),
            1 => "aborting due to previous error".to_string(),
            count => format!("aborting due to {count} previous errors"),
        };
        if self.treat_err_as_bug() {
            return;
        }

        match (errors.len(), warnings.len()) {
            (0, 0) => return,
            (0, _) => self.emitter.emit_diagnostic(&Diagnostic::new(
                Level::Warning(None),
                DiagnosticMessage::Str(warnings),
            )),
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

    fn stash(&mut self, key: (Span, StashKey), diagnostic: Diagnostic) {
        // Track the diagnostic for counts, but don't panic-if-treat-err-as-bug
        // yet; that happens when we actually emit the diagnostic.
        if diagnostic.is_error() {
            if matches!(diagnostic.level, Level::Error { lint: true }) {
                self.lint_err_count += 1;
            } else {
                self.err_count += 1;
            }
        } else {
            // Warnings are only automatically flushed if they're forced.
            if diagnostic.is_force_warn() {
                self.warn_count += 1;
            }
        }

        // FIXME(Centril, #69537): Consider reintroducing panic on overwriting a stashed diagnostic
        // if/when we have a more robust macro-friendly replacement for `(span, key)` as a key.
        // See the PR for a discussion.
        self.stashed_diagnostics.insert(key, diagnostic);
    }

    fn steal(&mut self, key: (Span, StashKey)) -> Option<Diagnostic> {
        let diagnostic = self.stashed_diagnostics.remove(&key)?;
        if diagnostic.is_error() {
            if matches!(diagnostic.level, Level::Error { lint: true }) {
                self.lint_err_count -= 1;
            } else {
                self.err_count -= 1;
            }
        } else {
            if diagnostic.is_force_warn() {
                self.warn_count -= 1;
            }
        }
        Some(diagnostic)
    }

    #[inline]
    fn err_count(&self) -> usize {
        self.err_count
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

    fn is_compilation_going_to_fail(&self) -> bool {
        self.has_errors() || self.lint_err_count > 0 || !self.delayed_span_bugs.is_empty()
    }

    fn abort_if_errors(&mut self) {
        self.emit_stashed_diagnostics();

        if self.has_errors() {
            FatalError.raise();
        }
    }

    #[track_caller]
    fn span_bug(&mut self, sp: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) -> ! {
        self.emit_diag_at_span(Diagnostic::new(Bug, msg), sp);
        panic::panic_any(ExplicitBug);
    }

    fn emit_diag_at_span(&mut self, mut diag: Diagnostic, sp: impl Into<MultiSpan>) {
        self.emit_diagnostic(diag.set_span(sp));
    }

    /// For documentation on this, see `Session::delay_span_bug`.
    #[track_caller]
    fn delay_span_bug(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        // This is technically `self.treat_err_as_bug()` but `delay_span_bug` is called before
        // incrementing `err_count` by one, so we need to +1 the comparing.
        // FIXME: Would be nice to increment err_count in a more coherent way.
        if self.flags.treat_err_as_bug.map_or(false, |c| {
            self.err_count() + self.lint_err_count + self.delayed_bug_count() + 1 >= c.get()
        }) {
            // FIXME: don't abort here if report_delayed_bugs is off
            self.span_bug(sp, msg);
        }
        let mut diagnostic = Diagnostic::new(Level::DelayedBug, msg);
        diagnostic.set_span(sp.into());
        self.emit_diagnostic(&mut diagnostic).unwrap()
    }

    // FIXME(eddyb) note the comment inside `impl Drop for HandlerInner`, that's
    // where the explanation of what "good path" is (also, it should be renamed).
    fn delay_good_path_bug(&mut self, msg: impl Into<DiagnosticMessage>) {
        let mut diagnostic = Diagnostic::new(Level::DelayedBug, msg);
        if self.flags.report_delayed_bugs {
            self.emit_diagnostic(&mut diagnostic);
        }
        let backtrace = std::backtrace::Backtrace::force_capture();
        self.delayed_good_path_bugs.push(DelayedDiagnostic::with_backtrace(diagnostic, backtrace));
    }

    fn failure(&mut self, msg: impl Into<DiagnosticMessage>) {
        self.emit_diagnostic(&mut Diagnostic::new(FailureNote, msg));
    }

    fn fatal(&mut self, msg: impl Into<DiagnosticMessage>) -> FatalError {
        self.emit(Fatal, msg);
        FatalError
    }

    fn err(&mut self, msg: impl Into<DiagnosticMessage>) -> ErrorGuaranteed {
        self.emit(Error { lint: false }, msg)
    }

    /// Emit an error; level should be `Error` or `Fatal`.
    fn emit(&mut self, level: Level, msg: impl Into<DiagnosticMessage>) -> ErrorGuaranteed {
        if self.treat_err_as_bug() {
            self.bug(msg);
        }
        self.emit_diagnostic(&mut Diagnostic::new(level, msg)).unwrap()
    }

    fn bug(&mut self, msg: impl Into<DiagnosticMessage>) -> ! {
        self.emit_diagnostic(&mut Diagnostic::new(Bug, msg));
        panic::panic_any(ExplicitBug);
    }

    fn flush_delayed(
        &mut self,
        bugs: impl IntoIterator<Item = DelayedDiagnostic>,
        explanation: impl Into<DiagnosticMessage> + Copy,
    ) {
        let mut no_bugs = true;
        for bug in bugs {
            let mut bug = bug.decorate();

            if no_bugs {
                // Put the overall explanation before the `DelayedBug`s, to
                // frame them better (e.g. separate warnings from them).
                self.emit_diagnostic(&mut Diagnostic::new(Bug, explanation));
                no_bugs = false;
            }

            // "Undelay" the `DelayedBug`s (into plain `Bug`s).
            if bug.level != Level::DelayedBug {
                // NOTE(eddyb) not panicking here because we're already producing
                // an ICE, and the more information the merrier.
                bug.note(&format!(
                    "`flushed_delayed` got diagnostic with level {:?}, \
                     instead of the expected `DelayedBug`",
                    bug.level,
                ));
            }
            bug.level = Level::Bug;

            self.emit_diagnostic(&mut bug);
        }

        // Panic with `DelayedBugPanic` to avoid "unexpected panic" messages.
        if !no_bugs {
            panic::panic_any(DelayedBugPanic);
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
                self.delayed_bug_count(),
                self.flags.treat_err_as_bug.map(|c| c.get()).unwrap_or(0),
            ) {
                (1, 0, 1) => panic!("aborting due to `-Z treat-err-as-bug=1`"),
                (0, 1, 1) => panic!("aborting due delayed bug with `-Z treat-err-as-bug=1`"),
                (count, delayed_count, as_bug) => {
                    if delayed_count > 0 {
                        panic!(
                            "aborting after {} errors and {} delayed bugs due to `-Z treat-err-as-bug={}`",
                            count, delayed_count, as_bug,
                        )
                    } else {
                        panic!(
                            "aborting after {} errors due to `-Z treat-err-as-bug={}`",
                            count, as_bug,
                        )
                    }
                }
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

#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug, Encodable, Decodable)]
pub enum Level {
    Bug,
    DelayedBug,
    Fatal,
    Error {
        /// If this error comes from a lint, don't abort compilation even when abort_if_errors() is called.
        lint: bool,
    },
    /// This [`LintExpectationId`] is used for expected lint diagnostics, which should
    /// also emit a warning due to the `force-warn` flag. In all other cases this should
    /// be `None`.
    Warning(Option<LintExpectationId>),
    Note,
    /// A note that is only emitted once.
    OnceNote,
    Help,
    FailureNote,
    Allow,
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
            Bug | DelayedBug | Fatal | Error { .. } => {
                spec.set_fg(Some(Color::Red)).set_intense(true);
            }
            Warning(_) => {
                spec.set_fg(Some(Color::Yellow)).set_intense(cfg!(windows));
            }
            Note | OnceNote => {
                spec.set_fg(Some(Color::Green)).set_intense(true);
            }
            Help => {
                spec.set_fg(Some(Color::Cyan)).set_intense(true);
            }
            FailureNote => {}
            Allow | Expect(_) => unreachable!(),
        }
        spec
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Bug | DelayedBug => "error: internal compiler error",
            Fatal | Error { .. } => "error",
            Warning(_) => "warning",
            Note | OnceNote => "note",
            Help => "help",
            FailureNote => "failure-note",
            Allow => panic!("Shouldn't call on allowed error"),
            Expect(_) => panic!("Shouldn't call on expected error"),
        }
    }

    pub fn is_failure_note(&self) -> bool {
        matches!(*self, FailureNote)
    }

    pub fn get_expectation_id(&self) -> Option<LintExpectationId> {
        match self {
            Level::Expect(id) | Level::Warning(Some(id)) => Some(*id),
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
    diag.span_label(path_span, format!("expected lifetime parameter{}", pluralize!(n)));
    if !source_map.is_span_accessible(insertion_span) {
        // Do not try to suggest anything if generated by a proc-macro.
        return;
    }
    let anon_lts = vec!["'_"; n].join(", ");
    let suggestion =
        if incl_angl_brckt { format!("<{}>", anon_lts) } else { format!("{}, ", anon_lts) };
    diag.span_suggestion_verbose(
        insertion_span.shrink_to_hi(),
        &format!("indicate the anonymous lifetime{}", pluralize!(n)),
        suggestion,
        Applicability::MachineApplicable,
    );
}

/// Useful type to use with `Result<>` indicate that an error has already
/// been reported to the user, so no need to continue checking.
#[derive(Clone, Copy, Debug, Encodable, Decodable, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[derive(HashStable_Generic)]
pub struct ErrorGuaranteed(());

impl ErrorGuaranteed {
    /// To be used only if you really know what you are doing... ideally, we would find a way to
    /// eliminate all calls to this method.
    pub fn unchecked_claim_error_was_emitted() -> Self {
        ErrorGuaranteed(())
    }
}
