//! Diagnostics creation and emission for `rustc`.
//!
//! This module contains the code for creating and emitting diagnostics.

// tidy-alphabetical-start
#![allow(incomplete_features)]
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![cfg_attr(not(bootstrap), allow(rustc::direct_use_of_rustc_type_ir))]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(array_windows)]
#![feature(assert_matches)]
#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(default_field_values)]
#![feature(error_reporter)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
// tidy-alphabetical-end

extern crate self as rustc_errors;

use std::assert_matches::assert_matches;
use std::backtrace::{Backtrace, BacktraceStatus};
use std::borrow::Cow;
use std::cell::Cell;
use std::error::Report;
use std::ffi::OsStr;
use std::hash::Hash;
use std::io::Write;
use std::num::NonZero;
use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::{fmt, panic};

use Level::*;
pub use codes::*;
pub use diagnostic::{
    BugAbort, Diag, DiagArg, DiagArgMap, DiagArgName, DiagArgValue, DiagInner, DiagStyledString,
    Diagnostic, EmissionGuarantee, FatalAbort, IntoDiagArg, LintDiagnostic, StringPart, Subdiag,
    Subdiagnostic,
};
pub use diagnostic_impls::{
    DiagArgFromDisplay, DiagSymbolList, ElidedLifetimeInPathSubdiag, ExpectedLifetimeParameter,
    IndicateAnonymousLifetime, SingleLabelManySpans,
};
pub use emitter::ColorConfig;
use emitter::{DynEmitter, Emitter, is_case_difference, is_different};
use rustc_data_structures::AtomicRef;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{DynSend, Lock};
pub use rustc_error_messages::{
    DiagMessage, FluentBundle, LanguageIdentifier, LazyFallbackBundle, MultiSpan, SpanLabel,
    SubdiagMessage, fallback_fluent_bundle, fluent_bundle,
};
use rustc_hashes::Hash128;
use rustc_hir::HirId;
pub use rustc_lint_defs::{Applicability, listify, pluralize};
use rustc_lint_defs::{Lint, LintExpectationId};
use rustc_macros::{Decodable, Encodable};
pub use rustc_span::ErrorGuaranteed;
pub use rustc_span::fatal_error::{FatalError, FatalErrorMarker};
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, DUMMY_SP, Loc, Span};
pub use snippet::Style;
// Used by external projects such as `rust-gpu`.
// See https://github.com/rust-lang/rust/pull/115393.
pub use termcolor::{Color, ColorSpec, WriteColor};
use tracing::debug;

use crate::emitter::TimingEvent;
use crate::registry::Registry;
use crate::timings::TimingRecord;

pub mod annotate_snippet_emitter_writer;
pub mod codes;
mod diagnostic;
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
pub mod timings;
pub mod translation;

pub type PResult<'a, T> = Result<T, Diag<'a>>;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

// `PResult` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(PResult<'_, ()>, 24);
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(PResult<'_, bool>, 24);

/// Used to avoid depending on `rustc_middle` in `rustc_attr_parsing`.
/// Always the `TyCtxt`.
pub trait LintEmitter: Copy {
    #[track_caller]
    fn emit_node_span_lint(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        decorator: impl for<'a> LintDiagnostic<'a, ()>,
    );
}

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

/// Represents the help messages seen on a diagnostic.
#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub enum Suggestions {
    /// Indicates that new suggestions can be added or removed from this diagnostic.
    ///
    /// `DiagInner`'s new_* methods initialize the `suggestions` field with
    /// this variant. Also, this is the default variant for `Suggestions`.
    Enabled(Vec<CodeSuggestion>),
    /// Indicates that suggestions cannot be added or removed from this diagnostic.
    ///
    /// Gets toggled when `.seal_suggestions()` is called on the `DiagInner`.
    Sealed(Box<[CodeSuggestion]>),
    /// Indicates that no suggestion is available for this diagnostic.
    ///
    /// Gets toggled when `.disable_suggestions()` is called on the `DiagInner`.
    Disabled,
}

impl Suggestions {
    /// Returns the underlying list of suggestions.
    pub fn unwrap_tag(self) -> Vec<CodeSuggestion> {
        match self {
            Suggestions::Enabled(suggestions) => suggestions,
            Suggestions::Sealed(suggestions) => suggestions.into_vec(),
            Suggestions::Disabled => Vec::new(),
        }
    }
}

impl Default for Suggestions {
    fn default() -> Self {
        Self::Enabled(vec![])
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
    pub msg: DiagMessage,
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

    /// Whether this is a replacement that overwrites source with a snippet
    /// in a way that isn't a superset of the original string. For example,
    /// replacing "abc" with "abcde" is not destructive, but replacing it
    /// it with "abx" is, since the "c" character is lost.
    pub fn is_destructive_replacement(&self, sm: &SourceMap) -> bool {
        self.is_replacement(sm)
            && !sm
                .span_to_snippet(self.span)
                .is_ok_and(|snippet| as_substr(snippet.trim(), self.snippet.trim()).is_some())
    }

    fn replaces_meaningful_content(&self, sm: &SourceMap) -> bool {
        sm.span_to_snippet(self.span)
            .map_or(!self.span.is_empty(), |snippet| !snippet.trim().is_empty())
    }

    /// Try to turn a replacement into an addition when the span that is being
    /// overwritten matches either the prefix or suffix of the replacement.
    fn trim_trivial_replacements(&mut self, sm: &SourceMap) {
        if self.snippet.is_empty() {
            return;
        }
        let Ok(snippet) = sm.span_to_snippet(self.span) else {
            return;
        };

        if let Some((prefix, substr, suffix)) = as_substr(&snippet, &self.snippet) {
            self.span = Span::new(
                self.span.lo() + BytePos(prefix as u32),
                self.span.hi() - BytePos(suffix as u32),
                self.span.ctxt(),
                self.span.parent(),
            );
            self.snippet = substr.to_string();
        }
    }
}

/// Given an original string like `AACC`, and a suggestion like `AABBCC`, try to detect
/// the case where a substring of the suggestion is "sandwiched" in the original, like
/// `BB` is. Return the length of the prefix, the "trimmed" suggestion, and the length
/// of the suffix.
fn as_substr<'a>(original: &'a str, suggestion: &'a str) -> Option<(usize, &'a str, usize)> {
    let common_prefix = original
        .chars()
        .zip(suggestion.chars())
        .take_while(|(c1, c2)| c1 == c2)
        .map(|(c, _)| c.len_utf8())
        .sum();
    let original = &original[common_prefix..];
    let suggestion = &suggestion[common_prefix..];
    if suggestion.ends_with(original) {
        let common_suffix = original.len();
        Some((common_prefix, &suggestion[..suggestion.len() - original.len()], common_suffix))
    } else {
        None
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

        /// Extracts a substring from the provided `line_opt` based on the specified low and high
        /// indices, appends it to the given buffer `buf`, and returns the count of newline
        /// characters in the substring for accurate highlighting. If `line_opt` is `None`, a
        /// newline character is appended to the buffer, and 0 is returned.
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
                let mut only_capitalization = false;
                for part in &mut substitution.parts {
                    // If this is a replacement of, e.g. `"a"` into `"ab"`, adjust the
                    // suggestion and snippet to look as if we just suggested to add
                    // `"b"`, which is typically much easier for the user to understand.
                    part.trim_trivial_replacements(sm);

                    only_capitalization |= is_case_difference(sm, &part.snippet, part.span);
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
                    if !is_different(sm, &part.snippet, part.span) {
                        // Account for cases where we are suggesting the same code that's already
                        // there. This shouldn't happen often, but in some cases for multipart
                        // suggestions it's much easier to handle it here than in the origin.
                    } else {
                        line_highlight.push(SubstitutionHighlight {
                            start: (cur_lo.col.0 as isize + acc) as usize,
                            end: (cur_lo.col.0 as isize + acc + len) as usize,
                        });
                    }
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
                // if the replacement already ends with a newline, don't print the next line
                if !buf.ends_with('\n') {
                    push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, None);
                }
                // remove trailing newlines
                while buf.ends_with('\n') {
                    buf.pop();
                }
                if highlights.iter().all(|parts| parts.is_empty()) {
                    None
                } else {
                    Some((buf, substitution.parts, highlights, only_capitalization))
                }
            })
            .collect()
    }
}

/// Signifies that the compiler died with an explicit call to `.bug`
/// or `.span_bug` rather than a failed assertion, etc.
pub struct ExplicitBug;

/// Signifies that the compiler died due to a delayed bug rather than a failed
/// assertion, etc.
pub struct DelayedBugPanic;

/// A `DiagCtxt` deals with errors and other compiler output.
/// Certain errors (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct DiagCtxt {
    inner: Lock<DiagCtxtInner>,
}

#[derive(Copy, Clone)]
pub struct DiagCtxtHandle<'a> {
    dcx: &'a DiagCtxt,
    /// Some contexts create `DiagCtxtHandle` with this field set, and thus all
    /// errors emitted with it will automatically taint when emitting errors.
    tainted_with_errors: Option<&'a Cell<Option<ErrorGuaranteed>>>,
}

impl<'a> std::ops::Deref for DiagCtxtHandle<'a> {
    type Target = &'a DiagCtxt;

    fn deref(&self) -> &Self::Target {
        &self.dcx
    }
}

/// This inner struct exists to keep it all behind a single lock;
/// this is done to prevent possible deadlocks in a multi-threaded compiler,
/// as well as inconsistent state observation.
struct DiagCtxtInner {
    flags: DiagCtxtFlags,

    registry: Registry,

    /// The error guarantees from all emitted errors. The length gives the error count.
    err_guars: Vec<ErrorGuaranteed>,
    /// The error guarantee from all emitted lint errors. The length gives the
    /// lint error count.
    lint_err_guars: Vec<ErrorGuaranteed>,
    /// The delayed bugs and their error guarantees.
    delayed_bugs: Vec<(DelayedDiagInner, ErrorGuaranteed)>,

    /// The error count shown to the user at the end.
    deduplicated_err_count: usize,
    /// The warning count shown to the user at the end.
    deduplicated_warn_count: usize,

    emitter: Box<DynEmitter>,

    /// Must we produce a diagnostic to justify the use of the expensive
    /// `trimmed_def_paths` function? Backtrace is the location of the call.
    must_produce_diag: Option<Backtrace>,

    /// Has this diagnostic context printed any diagnostics? (I.e. has
    /// `self.emitter.emit_diagnostic()` been called?
    has_printed: bool,

    /// This flag indicates that an expected diagnostic was emitted and suppressed.
    /// This is used for the `must_produce_diag` check.
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
    /// stolen and emitted/cancelled by other stages (e.g. to improve them and
    /// add more information). All stashed diagnostics must be emitted with
    /// `emit_stashed_diagnostics` by the time the `DiagCtxtInner` is dropped,
    /// otherwise an assertion failure will occur.
    stashed_diagnostics:
        FxIndexMap<StashKey, FxIndexMap<Span, (DiagInner, Option<ErrorGuaranteed>)>>,

    future_breakage_diagnostics: Vec<DiagInner>,

    /// expected diagnostic will have the level `Expect` which additionally
    /// carries the [`LintExpectationId`] of the expectation that can be
    /// marked as fulfilled. This is a collection of all [`LintExpectationId`]s
    /// that have been marked as fulfilled this way.
    ///
    /// Emitting expectations after having stolen this field can happen. In particular, an
    /// `#[expect(warnings)]` can easily make the `UNFULFILLED_LINT_EXPECTATIONS` lint expect
    /// itself. To avoid needless complexity in this corner case, we tolerate failing to track
    /// those expectations.
    ///
    /// [RFC-2383]: https://rust-lang.github.io/rfcs/2383-lint-reasons.html
    fulfilled_expectations: FxIndexSet<LintExpectationId>,

    /// The file where the ICE information is stored. This allows delayed_span_bug backtraces to be
    /// stored along side the main panic backtrace.
    ice_file: Option<PathBuf>,
}

/// A key denoting where from a diagnostic was stashed.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
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
    AssociatedTypeSuggestion,
    /// Query cycle detected, stashing in favor of a better error.
    Cycle,
    UndeterminedMacroResolution,
    /// Used by `Parser::maybe_recover_trailing_expr`
    ExprInPat,
    /// If in the parser we detect a field expr with turbofish generic params it's possible that
    /// it's a method call without parens. If later on in `hir_typeck` we find out that this is
    /// the case we suppress this message and we give a better suggestion.
    GenericInFieldExpr,
}

fn default_track_diagnostic<R>(diag: DiagInner, f: &mut dyn FnMut(DiagInner) -> R) -> R {
    (*f)(diag)
}

/// Diagnostics emitted by `DiagCtxtInner::emit_diagnostic` are passed through this function. Used
/// for tracking by incremental, to replay diagnostics as necessary.
pub static TRACK_DIAGNOSTIC: AtomicRef<
    fn(DiagInner, &mut dyn FnMut(DiagInner) -> Option<ErrorGuaranteed>) -> Option<ErrorGuaranteed>,
> = AtomicRef::new(&(default_track_diagnostic as _));

#[derive(Copy, Clone, Default)]
pub struct DiagCtxtFlags {
    /// If false, warning-level lints are suppressed.
    /// (rustc: see `--allow warnings` and `--cap-lints`)
    pub can_emit_warnings: bool,
    /// If Some, the Nth error-level diagnostic is upgraded to bug-level.
    /// (rustc: see `-Z treat-err-as-bug`)
    pub treat_err_as_bug: Option<NonZero<usize>>,
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
        // For tools using `interface::run_compiler` (e.g. rustc, rustdoc)
        // stashed diagnostics will have already been emitted. But for others
        // that don't use `interface::run_compiler` (e.g. rustfmt, some clippy
        // lints) this fallback is necessary.
        //
        // Important: it is sound to produce an `ErrorGuaranteed` when stashing
        // errors because they are guaranteed to be emitted here or earlier.
        self.emit_stashed_diagnostics();

        // Important: it is sound to produce an `ErrorGuaranteed` when emitting
        // delayed bugs because they are guaranteed to be emitted here if
        // necessary.
        self.flush_delayed();

        // Sanity check: did we use some of the expensive `trimmed_def_paths` functions
        // unexpectedly, that is, without producing diagnostics? If so, for debugging purposes, we
        // suggest where this happened and how to avoid it.
        if !self.has_printed && !self.suppressed_expected_diag && !std::thread::panicking() {
            if let Some(backtrace) = &self.must_produce_diag {
                let suggestion = match backtrace.status() {
                    BacktraceStatus::Disabled => String::from(
                        "Backtraces are currently disabled: set `RUST_BACKTRACE=1` and re-run \
                        to see where it happened.",
                    ),
                    BacktraceStatus::Captured => format!(
                        "This happened in the following `must_produce_diag` call's backtrace:\n\
                        {backtrace}",
                    ),
                    _ => String::from("(impossible to capture backtrace where this happened)"),
                };
                panic!(
                    "`trimmed_def_paths` called, diagnostics were expected but none were emitted. \
                    Use `with_no_trimmed_paths` for debugging. {suggestion}"
                );
            }
        }
    }
}

impl DiagCtxt {
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

    pub fn with_registry(mut self, registry: Registry) -> Self {
        self.inner.get_mut().registry = registry;
        self
    }

    pub fn new(emitter: Box<DynEmitter>) -> Self {
        Self { inner: Lock::new(DiagCtxtInner::new(emitter)) }
    }

    pub fn make_silent(&self, fatal_note: Option<String>, emit_fatal_diagnostic: bool) {
        // An empty type that implements `Emitter` to temporarily swap in place of the real one,
        // which will be used in constructing its replacement.
        struct FalseEmitter;

        impl Emitter for FalseEmitter {
            fn emit_diagnostic(&mut self, _: DiagInner, _: &Registry) {
                unimplemented!("false emitter must only used during `make_silent`")
            }

            fn source_map(&self) -> Option<&SourceMap> {
                unimplemented!("false emitter must only used during `make_silent`")
            }
        }

        impl translation::Translate for FalseEmitter {
            fn fluent_bundle(&self) -> Option<&FluentBundle> {
                unimplemented!("false emitter must only used during `make_silent`")
            }

            fn fallback_fluent_bundle(&self) -> &FluentBundle {
                unimplemented!("false emitter must only used during `make_silent`")
            }
        }

        let mut inner = self.inner.borrow_mut();
        let mut prev_emitter = Box::new(FalseEmitter) as Box<dyn Emitter + DynSend>;
        std::mem::swap(&mut inner.emitter, &mut prev_emitter);
        let new_emitter = Box::new(emitter::SilentEmitter {
            fatal_emitter: prev_emitter,
            fatal_note,
            emit_fatal_diagnostic,
        });
        inner.emitter = new_emitter;
    }

    pub fn set_emitter(&self, emitter: Box<dyn Emitter + DynSend>) {
        self.inner.borrow_mut().emitter = emitter;
    }

    /// Translate `message` eagerly with `args` to `SubdiagMessage::Eager`.
    pub fn eagerly_translate<'a>(
        &self,
        message: DiagMessage,
        args: impl Iterator<Item = DiagArg<'a>>,
    ) -> SubdiagMessage {
        let inner = self.inner.borrow();
        inner.eagerly_translate(message, args)
    }

    /// Translate `message` eagerly with `args` to `String`.
    pub fn eagerly_translate_to_string<'a>(
        &self,
        message: DiagMessage,
        args: impl Iterator<Item = DiagArg<'a>>,
    ) -> String {
        let inner = self.inner.borrow();
        inner.eagerly_translate_to_string(message, args)
    }

    // This is here to not allow mutation of flags;
    // as of this writing it's used in Session::consider_optimizing and
    // in tests in rustc_interface.
    pub fn can_emit_warnings(&self) -> bool {
        self.inner.borrow_mut().flags.can_emit_warnings
    }

    /// Resets the diagnostic error count as well as the cached emitted diagnostics.
    ///
    /// NOTE: *do not* call this function from rustc. It is only meant to be called from external
    /// tools that want to reuse a `Parser` cleaning the previously emitted diagnostics as well as
    /// the overall count of emitted error diagnostics.
    pub fn reset_err_count(&self) {
        // Use destructuring so that if a field gets added to `DiagCtxtInner`, it's impossible to
        // fail to update this method as well.
        let mut inner = self.inner.borrow_mut();
        let DiagCtxtInner {
            flags: _,
            registry: _,
            err_guars,
            lint_err_guars,
            delayed_bugs,
            deduplicated_err_count,
            deduplicated_warn_count,
            emitter: _,
            must_produce_diag,
            has_printed,
            suppressed_expected_diag,
            taught_diagnostics,
            emitted_diagnostic_codes,
            emitted_diagnostics,
            stashed_diagnostics,
            future_breakage_diagnostics,
            fulfilled_expectations,
            ice_file: _,
        } = inner.deref_mut();

        // For the `Vec`s and `HashMap`s, we overwrite with an empty container to free the
        // underlying memory (which `clear` would not do).
        *err_guars = Default::default();
        *lint_err_guars = Default::default();
        *delayed_bugs = Default::default();
        *deduplicated_err_count = 0;
        *deduplicated_warn_count = 0;
        *must_produce_diag = None;
        *has_printed = false;
        *suppressed_expected_diag = false;
        *taught_diagnostics = Default::default();
        *emitted_diagnostic_codes = Default::default();
        *emitted_diagnostics = Default::default();
        *stashed_diagnostics = Default::default();
        *future_breakage_diagnostics = Default::default();
        *fulfilled_expectations = Default::default();
    }

    pub fn handle<'a>(&'a self) -> DiagCtxtHandle<'a> {
        DiagCtxtHandle { dcx: self, tainted_with_errors: None }
    }

    /// Link this to a taintable context so that emitting errors will automatically set
    /// the `Option<ErrorGuaranteed>` instead of having to do that manually at every error
    /// emission site.
    pub fn taintable_handle<'a>(
        &'a self,
        tainted_with_errors: &'a Cell<Option<ErrorGuaranteed>>,
    ) -> DiagCtxtHandle<'a> {
        DiagCtxtHandle { dcx: self, tainted_with_errors: Some(tainted_with_errors) }
    }
}

impl<'a> DiagCtxtHandle<'a> {
    /// Stashes a diagnostic for possible later improvement in a different,
    /// later stage of the compiler. Possible actions depend on the diagnostic
    /// level:
    /// - Level::Bug, Level:Fatal: not allowed, will trigger a panic.
    /// - Level::Error: immediately counted as an error that has occurred, because it
    ///   is guaranteed to be emitted eventually. Can be later accessed with the
    ///   provided `span` and `key` through
    ///   [`DiagCtxtHandle::try_steal_modify_and_emit_err`] or
    ///   [`DiagCtxtHandle::try_steal_replace_and_emit_err`]. These do not allow
    ///   cancellation or downgrading of the error. Returns
    ///   `Some(ErrorGuaranteed)`.
    /// - Level::DelayedBug: this does happen occasionally with errors that are
    ///   downgraded to delayed bugs. It is not stashed, but immediately
    ///   emitted as a delayed bug. This is because stashing it would cause it
    ///   to be counted by `err_count` which we don't want. It doesn't matter
    ///   that we cannot steal and improve it later, because it's not a
    ///   user-facing error. Returns `Some(ErrorGuaranteed)` as is normal for
    ///   delayed bugs.
    /// - Level::Warning and lower (i.e. !is_error()): can be accessed with the
    ///   provided `span` and `key` through [`DiagCtxtHandle::steal_non_err()`]. This
    ///   allows cancelling and downgrading of the diagnostic. Returns `None`.
    pub fn stash_diagnostic(
        &self,
        span: Span,
        key: StashKey,
        diag: DiagInner,
    ) -> Option<ErrorGuaranteed> {
        let guar = match diag.level {
            Bug | Fatal => {
                self.span_bug(
                    span,
                    format!("invalid level in `stash_diagnostic`: {:?}", diag.level),
                );
            }
            // We delay a bug here so that `-Ztreat-err-as-bug -Zeagerly-emit-delayed-bugs`
            // can be used to create a backtrace at the stashing site instead of whenever the
            // diagnostic context is dropped and thus delayed bugs are emitted.
            Error => Some(self.span_delayed_bug(span, format!("stashing {key:?}"))),
            DelayedBug => {
                return self.inner.borrow_mut().emit_diagnostic(diag, self.tainted_with_errors);
            }
            ForceWarning | Warning | Note | OnceNote | Help | OnceHelp | FailureNote | Allow
            | Expect => None,
        };

        // FIXME(Centril, #69537): Consider reintroducing panic on overwriting a stashed diagnostic
        // if/when we have a more robust macro-friendly replacement for `(span, key)` as a key.
        // See the PR for a discussion.
        self.inner
            .borrow_mut()
            .stashed_diagnostics
            .entry(key)
            .or_default()
            .insert(span.with_parent(None), (diag, guar));

        guar
    }

    /// Steal a previously stashed non-error diagnostic with the given `Span`
    /// and [`StashKey`] as the key. Panics if the found diagnostic is an
    /// error.
    pub fn steal_non_err(self, span: Span, key: StashKey) -> Option<Diag<'a, ()>> {
        // FIXME(#120456) - is `swap_remove` correct?
        let (diag, guar) = self.inner.borrow_mut().stashed_diagnostics.get_mut(&key).and_then(
            |stashed_diagnostics| stashed_diagnostics.swap_remove(&span.with_parent(None)),
        )?;
        assert!(!diag.is_error());
        assert!(guar.is_none());
        Some(Diag::new_diagnostic(self, diag))
    }

    /// Steals a previously stashed error with the given `Span` and
    /// [`StashKey`] as the key, modifies it, and emits it. Returns `None` if
    /// no matching diagnostic is found. Panics if the found diagnostic's level
    /// isn't `Level::Error`.
    pub fn try_steal_modify_and_emit_err<F>(
        self,
        span: Span,
        key: StashKey,
        mut modify_err: F,
    ) -> Option<ErrorGuaranteed>
    where
        F: FnMut(&mut Diag<'_>),
    {
        // FIXME(#120456) - is `swap_remove` correct?
        let err = self.inner.borrow_mut().stashed_diagnostics.get_mut(&key).and_then(
            |stashed_diagnostics| stashed_diagnostics.swap_remove(&span.with_parent(None)),
        );
        err.map(|(err, guar)| {
            // The use of `::<ErrorGuaranteed>` is safe because level is `Level::Error`.
            assert_eq!(err.level, Error);
            assert!(guar.is_some());
            let mut err = Diag::<ErrorGuaranteed>::new_diagnostic(self, err);
            modify_err(&mut err);
            assert_eq!(err.level, Error);
            err.emit()
        })
    }

    /// Steals a previously stashed error with the given `Span` and
    /// [`StashKey`] as the key, cancels it if found, and emits `new_err`.
    /// Panics if the found diagnostic's level isn't `Level::Error`.
    pub fn try_steal_replace_and_emit_err(
        self,
        span: Span,
        key: StashKey,
        new_err: Diag<'_>,
    ) -> ErrorGuaranteed {
        // FIXME(#120456) - is `swap_remove` correct?
        let old_err = self.inner.borrow_mut().stashed_diagnostics.get_mut(&key).and_then(
            |stashed_diagnostics| stashed_diagnostics.swap_remove(&span.with_parent(None)),
        );
        match old_err {
            Some((old_err, guar)) => {
                assert_eq!(old_err.level, Error);
                assert!(guar.is_some());
                // Because `old_err` has already been counted, it can only be
                // safely cancelled because the `new_err` supplants it.
                Diag::<ErrorGuaranteed>::new_diagnostic(self, old_err).cancel();
            }
            None => {}
        };
        new_err.emit()
    }

    pub fn has_stashed_diagnostic(&self, span: Span, key: StashKey) -> bool {
        let inner = self.inner.borrow();
        if let Some(stashed_diagnostics) = inner.stashed_diagnostics.get(&key)
            && !stashed_diagnostics.is_empty()
        {
            stashed_diagnostics.contains_key(&span.with_parent(None))
        } else {
            false
        }
    }

    /// Emit all stashed diagnostics.
    pub fn emit_stashed_diagnostics(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow_mut().emit_stashed_diagnostics()
    }

    /// This excludes delayed bugs.
    #[inline]
    pub fn err_count(&self) -> usize {
        let inner = self.inner.borrow();
        inner.err_guars.len()
            + inner.lint_err_guars.len()
            + inner
                .stashed_diagnostics
                .values()
                .map(|a| a.values().filter(|(_, guar)| guar.is_some()).count())
                .sum::<usize>()
    }

    /// This excludes lint errors and delayed bugs. Unless absolutely
    /// necessary, prefer `has_errors` to this method.
    pub fn has_errors_excluding_lint_errors(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow().has_errors_excluding_lint_errors()
    }

    /// This excludes delayed bugs.
    pub fn has_errors(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow().has_errors()
    }

    /// This excludes nothing. Unless absolutely necessary, prefer `has_errors`
    /// to this method.
    pub fn has_errors_or_delayed_bugs(&self) -> Option<ErrorGuaranteed> {
        self.inner.borrow().has_errors_or_delayed_bugs()
    }

    pub fn print_error_count(&self) {
        let mut inner = self.inner.borrow_mut();

        // Any stashed diagnostics should have been handled by
        // `emit_stashed_diagnostics` by now.
        assert!(inner.stashed_diagnostics.is_empty());

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
            (0, _) => {
                // Use `ForceWarning` rather than `Warning` to guarantee emission, e.g. with a
                // configuration like `--cap-lints allow --force-warn bare_trait_objects`.
                inner.emit_diagnostic(
                    DiagInner::new(ForceWarning, DiagMessage::Str(warnings)),
                    None,
                );
            }
            (_, 0) => {
                inner.emit_diagnostic(DiagInner::new(Error, errors), self.tainted_with_errors);
            }
            (_, _) => {
                inner.emit_diagnostic(
                    DiagInner::new(Error, format!("{errors}; {warnings}")),
                    self.tainted_with_errors,
                );
            }
        }

        let can_show_explain = inner.emitter.should_show_explain();
        let are_there_diagnostics = !inner.emitted_diagnostic_codes.is_empty();
        if can_show_explain && are_there_diagnostics {
            let mut error_codes = inner
                .emitted_diagnostic_codes
                .iter()
                .filter_map(|&code| {
                    if inner.registry.try_find_description(code).is_ok() {
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
                    let msg1 = format!(
                        "Some errors have detailed explanations: {}{}",
                        error_codes[..limit].join(", "),
                        if error_codes.len() > 9 { "..." } else { "." }
                    );
                    let msg2 = format!(
                        "For more information about an error, try `rustc --explain {}`.",
                        &error_codes[0]
                    );
                    inner.emit_diagnostic(DiagInner::new(FailureNote, msg1), None);
                    inner.emit_diagnostic(DiagInner::new(FailureNote, msg2), None);
                } else {
                    let msg = format!(
                        "For more information about this error, try `rustc --explain {}`.",
                        &error_codes[0]
                    );
                    inner.emit_diagnostic(DiagInner::new(FailureNote, msg), None);
                }
            }
        }
    }

    /// This excludes delayed bugs. Used for early aborts after errors occurred
    /// -- e.g. because continuing in the face of errors is likely to lead to
    /// bad results, such as spurious/uninteresting additional errors -- when
    /// returning an error `Result` is difficult.
    pub fn abort_if_errors(&self) {
        if let Some(guar) = self.has_errors() {
            guar.raise_fatal();
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

    pub fn emit_diagnostic(&self, diagnostic: DiagInner) -> Option<ErrorGuaranteed> {
        self.inner.borrow_mut().emit_diagnostic(diagnostic, self.tainted_with_errors)
    }

    pub fn emit_artifact_notification(&self, path: &Path, artifact_type: &str) {
        self.inner.borrow_mut().emitter.emit_artifact_notification(path, artifact_type);
    }

    pub fn emit_timing_section_start(&self, record: TimingRecord) {
        self.inner.borrow_mut().emitter.emit_timing_section(record, TimingEvent::Start);
    }

    pub fn emit_timing_section_end(&self, record: TimingRecord) {
        self.inner.borrow_mut().emitter.emit_timing_section(record, TimingEvent::End);
    }

    pub fn emit_future_breakage_report(&self) {
        let inner = &mut *self.inner.borrow_mut();
        let diags = std::mem::take(&mut inner.future_breakage_diagnostics);
        if !diags.is_empty() {
            inner.emitter.emit_future_breakage_report(diags, &inner.registry);
        }
    }

    pub fn emit_unused_externs(
        &self,
        lint_level: rustc_lint_defs::Level,
        loud: bool,
        unused_externs: &[&str],
    ) {
        let mut inner = self.inner.borrow_mut();

        // This "error" is an odd duck.
        // - It's only produce with JSON output.
        // - It's not emitted the usual way, via `emit_diagnostic`.
        // - The `$message_type` field is "unused_externs" rather than the usual
        //   "diagnosic".
        //
        // We count it as a lint error because it has a lint level. The value
        // of `loud` (which comes from "unused-externs" or
        // "unused-externs-silent"), also affects whether it's treated like a
        // hard error or not.
        if loud && lint_level.is_error() {
            // This `unchecked_error_guaranteed` is valid. It is where the
            // `ErrorGuaranteed` for unused_extern errors originates.
            #[allow(deprecated)]
            inner.lint_err_guars.push(ErrorGuaranteed::unchecked_error_guaranteed());
            inner.panic_if_treat_err_as_bug();
        }

        inner.emitter.emit_unused_externs(lint_level, unused_externs)
    }

    /// This methods steals all [`LintExpectationId`]s that are stored inside
    /// [`DiagCtxtInner`] and indicate that the linked expectation has been fulfilled.
    #[must_use]
    pub fn steal_fulfilled_expectation_ids(&self) -> FxIndexSet<LintExpectationId> {
        std::mem::take(&mut self.inner.borrow_mut().fulfilled_expectations)
    }

    pub fn flush_delayed(&self) {
        self.inner.borrow_mut().flush_delayed();
    }

    /// Used when trimmed_def_paths is called and we must produce a diagnostic
    /// to justify its cost.
    #[track_caller]
    pub fn set_must_produce_diag(&self) {
        assert!(
            self.inner.borrow().must_produce_diag.is_none(),
            "should only need to collect a backtrace once"
        );
        self.inner.borrow_mut().must_produce_diag = Some(Backtrace::capture());
    }
}

// This `impl` block contains only the public diagnostic creation/emission API.
//
// Functions beginning with `struct_`/`create_` create a diagnostic. Other
// functions create and emit a diagnostic all in one go.
impl<'a> DiagCtxtHandle<'a> {
    // No `#[rustc_lint_diagnostics]` and no `impl Into<DiagMessage>` because bug messages aren't
    // user-facing.
    #[track_caller]
    pub fn struct_bug(self, msg: impl Into<Cow<'static, str>>) -> Diag<'a, BugAbort> {
        Diag::new(self, Bug, msg.into())
    }

    // No `#[rustc_lint_diagnostics]` and no `impl Into<DiagMessage>` because bug messages aren't
    // user-facing.
    #[track_caller]
    pub fn bug(self, msg: impl Into<Cow<'static, str>>) -> ! {
        self.struct_bug(msg).emit()
    }

    // No `#[rustc_lint_diagnostics]` and no `impl Into<DiagMessage>` because bug messages aren't
    // user-facing.
    #[track_caller]
    pub fn struct_span_bug(
        self,
        span: impl Into<MultiSpan>,
        msg: impl Into<Cow<'static, str>>,
    ) -> Diag<'a, BugAbort> {
        self.struct_bug(msg).with_span(span)
    }

    // No `#[rustc_lint_diagnostics]` and no `impl Into<DiagMessage>` because bug messages aren't
    // user-facing.
    #[track_caller]
    pub fn span_bug(self, span: impl Into<MultiSpan>, msg: impl Into<Cow<'static, str>>) -> ! {
        self.struct_span_bug(span, msg.into()).emit()
    }

    #[track_caller]
    pub fn create_bug(self, bug: impl Diagnostic<'a, BugAbort>) -> Diag<'a, BugAbort> {
        bug.into_diag(self, Bug)
    }

    #[track_caller]
    pub fn emit_bug(self, bug: impl Diagnostic<'a, BugAbort>) -> ! {
        self.create_bug(bug).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_fatal(self, msg: impl Into<DiagMessage>) -> Diag<'a, FatalAbort> {
        Diag::new(self, Fatal, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn fatal(self, msg: impl Into<DiagMessage>) -> ! {
        self.struct_fatal(msg).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_fatal(
        self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagMessage>,
    ) -> Diag<'a, FatalAbort> {
        self.struct_fatal(msg).with_span(span)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_fatal(self, span: impl Into<MultiSpan>, msg: impl Into<DiagMessage>) -> ! {
        self.struct_span_fatal(span, msg).emit()
    }

    #[track_caller]
    pub fn create_fatal(self, fatal: impl Diagnostic<'a, FatalAbort>) -> Diag<'a, FatalAbort> {
        fatal.into_diag(self, Fatal)
    }

    #[track_caller]
    pub fn emit_fatal(self, fatal: impl Diagnostic<'a, FatalAbort>) -> ! {
        self.create_fatal(fatal).emit()
    }

    #[track_caller]
    pub fn create_almost_fatal(
        self,
        fatal: impl Diagnostic<'a, FatalError>,
    ) -> Diag<'a, FatalError> {
        fatal.into_diag(self, Fatal)
    }

    #[track_caller]
    pub fn emit_almost_fatal(self, fatal: impl Diagnostic<'a, FatalError>) -> FatalError {
        self.create_almost_fatal(fatal).emit()
    }

    // FIXME: This method should be removed (every error should have an associated error code).
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_err(self, msg: impl Into<DiagMessage>) -> Diag<'a> {
        Diag::new(self, Error, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn err(self, msg: impl Into<DiagMessage>) -> ErrorGuaranteed {
        self.struct_err(msg).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_err(
        self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagMessage>,
    ) -> Diag<'a> {
        self.struct_err(msg).with_span(span)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_err(
        self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagMessage>,
    ) -> ErrorGuaranteed {
        self.struct_span_err(span, msg).emit()
    }

    #[track_caller]
    pub fn create_err(self, err: impl Diagnostic<'a>) -> Diag<'a> {
        err.into_diag(self, Error)
    }

    #[track_caller]
    pub fn emit_err(self, err: impl Diagnostic<'a>) -> ErrorGuaranteed {
        self.create_err(err).emit()
    }

    /// Ensures that an error is printed. See `Level::DelayedBug`.
    //
    // No `#[rustc_lint_diagnostics]` and no `impl Into<DiagMessage>` because bug messages aren't
    // user-facing.
    #[track_caller]
    pub fn delayed_bug(self, msg: impl Into<Cow<'static, str>>) -> ErrorGuaranteed {
        Diag::<ErrorGuaranteed>::new(self, DelayedBug, msg.into()).emit()
    }

    /// Ensures that an error is printed. See [`Level::DelayedBug`].
    ///
    /// Note: this function used to be called `delay_span_bug`. It was renamed
    /// to match similar functions like `span_err`, `span_warn`, etc.
    //
    // No `#[rustc_lint_diagnostics]` and no `impl Into<DiagMessage>` because bug messages aren't
    // user-facing.
    #[track_caller]
    pub fn span_delayed_bug(
        self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<Cow<'static, str>>,
    ) -> ErrorGuaranteed {
        Diag::<ErrorGuaranteed>::new(self, DelayedBug, msg.into()).with_span(sp).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn(self, msg: impl Into<DiagMessage>) -> Diag<'a, ()> {
        Diag::new(self, Warning, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn warn(self, msg: impl Into<DiagMessage>) {
        self.struct_warn(msg).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn(
        self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagMessage>,
    ) -> Diag<'a, ()> {
        self.struct_warn(msg).with_span(span)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_warn(self, span: impl Into<MultiSpan>, msg: impl Into<DiagMessage>) {
        self.struct_span_warn(span, msg).emit()
    }

    #[track_caller]
    pub fn create_warn(self, warning: impl Diagnostic<'a, ()>) -> Diag<'a, ()> {
        warning.into_diag(self, Warning)
    }

    #[track_caller]
    pub fn emit_warn(self, warning: impl Diagnostic<'a, ()>) {
        self.create_warn(warning).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_note(self, msg: impl Into<DiagMessage>) -> Diag<'a, ()> {
        Diag::new(self, Note, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn note(&self, msg: impl Into<DiagMessage>) {
        self.struct_note(msg).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_note(
        self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagMessage>,
    ) -> Diag<'a, ()> {
        self.struct_note(msg).with_span(span)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_note(self, span: impl Into<MultiSpan>, msg: impl Into<DiagMessage>) {
        self.struct_span_note(span, msg).emit()
    }

    #[track_caller]
    pub fn create_note(self, note: impl Diagnostic<'a, ()>) -> Diag<'a, ()> {
        note.into_diag(self, Note)
    }

    #[track_caller]
    pub fn emit_note(self, note: impl Diagnostic<'a, ()>) {
        self.create_note(note).emit()
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_help(self, msg: impl Into<DiagMessage>) -> Diag<'a, ()> {
        Diag::new(self, Help, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_failure_note(self, msg: impl Into<DiagMessage>) -> Diag<'a, ()> {
        Diag::new(self, FailureNote, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_allow(self, msg: impl Into<DiagMessage>) -> Diag<'a, ()> {
        Diag::new(self, Allow, msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_expect(self, msg: impl Into<DiagMessage>, id: LintExpectationId) -> Diag<'a, ()> {
        Diag::new(self, Expect, msg).with_lint_id(id)
    }
}

// Note: we prefer implementing operations on `DiagCtxt`, rather than
// `DiagCtxtInner`, whenever possible. This minimizes functions where
// `DiagCtxt::foo()` just borrows `inner` and forwards a call to
// `DiagCtxtInner::foo`.
impl DiagCtxtInner {
    fn new(emitter: Box<DynEmitter>) -> Self {
        Self {
            flags: DiagCtxtFlags { can_emit_warnings: true, ..Default::default() },
            registry: Registry::new(&[]),
            err_guars: Vec::new(),
            lint_err_guars: Vec::new(),
            delayed_bugs: Vec::new(),
            deduplicated_err_count: 0,
            deduplicated_warn_count: 0,
            emitter,
            must_produce_diag: None,
            has_printed: false,
            suppressed_expected_diag: false,
            taught_diagnostics: Default::default(),
            emitted_diagnostic_codes: Default::default(),
            emitted_diagnostics: Default::default(),
            stashed_diagnostics: Default::default(),
            future_breakage_diagnostics: Vec::new(),
            fulfilled_expectations: Default::default(),
            ice_file: None,
        }
    }

    /// Emit all stashed diagnostics.
    fn emit_stashed_diagnostics(&mut self) -> Option<ErrorGuaranteed> {
        let mut guar = None;
        let has_errors = !self.err_guars.is_empty();
        for (_, stashed_diagnostics) in std::mem::take(&mut self.stashed_diagnostics).into_iter() {
            for (_, (diag, _guar)) in stashed_diagnostics {
                if !diag.is_error() {
                    // Unless they're forced, don't flush stashed warnings when
                    // there are errors, to avoid causing warning overload. The
                    // stash would've been stolen already if it were important.
                    if !diag.is_force_warn() && has_errors {
                        continue;
                    }
                }
                guar = guar.or(self.emit_diagnostic(diag, None));
            }
        }
        guar
    }

    // Return value is only `Some` if the level is `Error` or `DelayedBug`.
    fn emit_diagnostic(
        &mut self,
        mut diagnostic: DiagInner,
        taint: Option<&Cell<Option<ErrorGuaranteed>>>,
    ) -> Option<ErrorGuaranteed> {
        if diagnostic.has_future_breakage() {
            // Future breakages aren't emitted if they're `Level::Allow` or
            // `Level::Expect`, but they still need to be constructed and
            // stashed below, so they'll trigger the must_produce_diag check.
            assert_matches!(diagnostic.level, Error | ForceWarning | Warning | Allow | Expect);
            self.future_breakage_diagnostics.push(diagnostic.clone());
        }

        // We call TRACK_DIAGNOSTIC with an empty closure for the cases that
        // return early *and* have some kind of side-effect, except where
        // noted.
        match diagnostic.level {
            Bug => {}
            Fatal | Error => {
                if self.treat_next_err_as_bug() {
                    // `Fatal` and `Error` can be promoted to `Bug`.
                    diagnostic.level = Bug;
                }
            }
            DelayedBug => {
                // Note that because we check these conditions first,
                // `-Zeagerly-emit-delayed-bugs` and `-Ztreat-err-as-bug`
                // continue to work even after we've issued an error and
                // stopped recording new delayed bugs.
                if self.flags.eagerly_emit_delayed_bugs {
                    // `DelayedBug` can be promoted to `Error` or `Bug`.
                    if self.treat_next_err_as_bug() {
                        diagnostic.level = Bug;
                    } else {
                        diagnostic.level = Error;
                    }
                } else {
                    // If we have already emitted at least one error, we don't need
                    // to record the delayed bug, because it'll never be used.
                    return if let Some(guar) = self.has_errors() {
                        Some(guar)
                    } else {
                        // No `TRACK_DIAGNOSTIC` call is needed, because the
                        // incremental session is deleted if there is a delayed
                        // bug. This also saves us from cloning the diagnostic.
                        let backtrace = std::backtrace::Backtrace::capture();
                        // This `unchecked_error_guaranteed` is valid. It is where the
                        // `ErrorGuaranteed` for delayed bugs originates. See
                        // `DiagCtxtInner::drop`.
                        #[allow(deprecated)]
                        let guar = ErrorGuaranteed::unchecked_error_guaranteed();
                        self.delayed_bugs
                            .push((DelayedDiagInner::with_backtrace(diagnostic, backtrace), guar));
                        Some(guar)
                    };
                }
            }
            ForceWarning if diagnostic.lint_id.is_none() => {} // `ForceWarning(Some(...))` is below, with `Expect`
            Warning => {
                if !self.flags.can_emit_warnings {
                    // We are not emitting warnings.
                    if diagnostic.has_future_breakage() {
                        // The side-effect is at the top of this method.
                        TRACK_DIAGNOSTIC(diagnostic, &mut |_| None);
                    }
                    return None;
                }
            }
            Note | Help | FailureNote => {}
            OnceNote | OnceHelp => panic!("bad level: {:?}", diagnostic.level),
            Allow => {
                // Nothing emitted for allowed lints.
                if diagnostic.has_future_breakage() {
                    // The side-effect is at the top of this method.
                    TRACK_DIAGNOSTIC(diagnostic, &mut |_| None);
                    self.suppressed_expected_diag = true;
                }
                return None;
            }
            Expect | ForceWarning => {
                self.fulfilled_expectations.insert(diagnostic.lint_id.unwrap());
                if let Expect = diagnostic.level {
                    // Nothing emitted here for expected lints.
                    TRACK_DIAGNOSTIC(diagnostic, &mut |_| None);
                    self.suppressed_expected_diag = true;
                    return None;
                }
            }
        }

        TRACK_DIAGNOSTIC(diagnostic, &mut |mut diagnostic| {
            if let Some(code) = diagnostic.code {
                self.emitted_diagnostic_codes.insert(code);
            }

            let already_emitted = {
                let mut hasher = StableHasher::new();
                diagnostic.hash(&mut hasher);
                let diagnostic_hash = hasher.finish();
                !self.emitted_diagnostics.insert(diagnostic_hash)
            };

            let is_error = diagnostic.is_error();
            let is_lint = diagnostic.is_lint.is_some();

            // Only emit the diagnostic if we've been asked to deduplicate or
            // haven't already emitted an equivalent diagnostic.
            if !(self.flags.deduplicate_diagnostics && already_emitted) {
                debug!(?diagnostic);
                debug!(?self.emitted_diagnostics);

                let not_yet_emitted = |sub: &mut Subdiag| {
                    debug!(?sub);
                    if sub.level != OnceNote && sub.level != OnceHelp {
                        return true;
                    }
                    let mut hasher = StableHasher::new();
                    sub.hash(&mut hasher);
                    let diagnostic_hash = hasher.finish();
                    debug!(?diagnostic_hash);
                    self.emitted_diagnostics.insert(diagnostic_hash)
                };
                diagnostic.children.retain_mut(not_yet_emitted);
                if already_emitted {
                    let msg = "duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`";
                    diagnostic.sub(Note, msg, MultiSpan::new());
                }

                if is_error {
                    self.deduplicated_err_count += 1;
                } else if matches!(diagnostic.level, ForceWarning | Warning) {
                    self.deduplicated_warn_count += 1;
                }
                self.has_printed = true;

                self.emitter.emit_diagnostic(diagnostic, &self.registry);
            }

            if is_error {
                // If we have any delayed bugs recorded, we can discard them
                // because they won't be used. (This should only occur if there
                // have been no errors previously emitted, because we don't add
                // new delayed bugs once the first error is emitted.)
                if !self.delayed_bugs.is_empty() {
                    assert_eq!(self.lint_err_guars.len() + self.err_guars.len(), 0);
                    self.delayed_bugs.clear();
                    self.delayed_bugs.shrink_to_fit();
                }

                // This `unchecked_error_guaranteed` is valid. It is where the
                // `ErrorGuaranteed` for errors and lint errors originates.
                #[allow(deprecated)]
                let guar = ErrorGuaranteed::unchecked_error_guaranteed();
                if is_lint {
                    self.lint_err_guars.push(guar);
                } else {
                    if let Some(taint) = taint {
                        taint.set(Some(guar));
                    }
                    self.err_guars.push(guar);
                }
                self.panic_if_treat_err_as_bug();
                Some(guar)
            } else {
                None
            }
        })
    }

    fn treat_err_as_bug(&self) -> bool {
        self.flags
            .treat_err_as_bug
            .is_some_and(|c| self.err_guars.len() + self.lint_err_guars.len() >= c.get())
    }

    // Use this one before incrementing `err_count`.
    fn treat_next_err_as_bug(&self) -> bool {
        self.flags
            .treat_err_as_bug
            .is_some_and(|c| self.err_guars.len() + self.lint_err_guars.len() + 1 >= c.get())
    }

    fn has_errors_excluding_lint_errors(&self) -> Option<ErrorGuaranteed> {
        self.err_guars.get(0).copied().or_else(|| {
            if let Some((_diag, guar)) = self
                .stashed_diagnostics
                .values()
                .flat_map(|stashed_diagnostics| stashed_diagnostics.values())
                .find(|(diag, guar)| guar.is_some() && diag.is_lint.is_none())
            {
                *guar
            } else {
                None
            }
        })
    }

    fn has_errors(&self) -> Option<ErrorGuaranteed> {
        self.err_guars.get(0).copied().or_else(|| self.lint_err_guars.get(0).copied()).or_else(
            || {
                self.stashed_diagnostics.values().find_map(|stashed_diagnostics| {
                    stashed_diagnostics.values().find_map(|(_, guar)| *guar)
                })
            },
        )
    }

    fn has_errors_or_delayed_bugs(&self) -> Option<ErrorGuaranteed> {
        self.has_errors().or_else(|| self.delayed_bugs.get(0).map(|(_, guar)| guar).copied())
    }

    /// Translate `message` eagerly with `args` to `SubdiagMessage::Eager`.
    fn eagerly_translate<'a>(
        &self,
        message: DiagMessage,
        args: impl Iterator<Item = DiagArg<'a>>,
    ) -> SubdiagMessage {
        SubdiagMessage::Translated(Cow::from(self.eagerly_translate_to_string(message, args)))
    }

    /// Translate `message` eagerly with `args` to `String`.
    fn eagerly_translate_to_string<'a>(
        &self,
        message: DiagMessage,
        args: impl Iterator<Item = DiagArg<'a>>,
    ) -> String {
        let args = crate::translation::to_fluent_args(args);
        self.emitter.translate_message(&message, &args).map_err(Report::new).unwrap().to_string()
    }

    fn eagerly_translate_for_subdiag(
        &self,
        diag: &DiagInner,
        msg: impl Into<SubdiagMessage>,
    ) -> SubdiagMessage {
        let msg = diag.subdiagnostic_message_to_diagnostic_message(msg);
        self.eagerly_translate(msg, diag.args.iter())
    }

    fn flush_delayed(&mut self) {
        // Stashed diagnostics must be emitted before delayed bugs are flushed.
        // Otherwise, we might ICE prematurely when errors would have
        // eventually happened.
        assert!(self.stashed_diagnostics.is_empty());

        if !self.err_guars.is_empty() {
            // If an error happened already. We shouldn't expose delayed bugs.
            return;
        }

        if self.delayed_bugs.is_empty() {
            // Nothing to do.
            return;
        }

        let bugs: Vec<_> =
            std::mem::take(&mut self.delayed_bugs).into_iter().map(|(b, _)| b).collect();

        let backtrace = std::env::var_os("RUST_BACKTRACE").as_deref() != Some(OsStr::new("0"));
        let decorate = backtrace || self.ice_file.is_none();
        let mut out = self
            .ice_file
            .as_ref()
            .and_then(|file| std::fs::File::options().create(true).append(true).open(file).ok());

        // Put the overall explanation before the `DelayedBug`s, to frame them
        // better (e.g. separate warnings from them). Also, use notes, which
        // don't count as errors, to avoid possibly triggering
        // `-Ztreat-err-as-bug`, which we don't want.
        let note1 = "no errors encountered even though delayed bugs were created";
        let note2 = "those delayed bugs will now be shown as internal compiler errors";
        self.emit_diagnostic(DiagInner::new(Note, note1), None);
        self.emit_diagnostic(DiagInner::new(Note, note2), None);

        for bug in bugs {
            if let Some(out) = &mut out {
                _ = write!(
                    out,
                    "delayed bug: {}\n{}\n",
                    bug.inner
                        .messages
                        .iter()
                        .filter_map(|(msg, _)| msg.as_str())
                        .collect::<String>(),
                    &bug.note
                );
            }

            let mut bug = if decorate { bug.decorate(self) } else { bug.inner };

            // "Undelay" the delayed bugs into plain bugs.
            if bug.level != DelayedBug {
                // NOTE(eddyb) not panicking here because we're already producing
                // an ICE, and the more information the merrier.
                //
                // We are at the `DiagInner`/`DiagCtxtInner` level rather than
                // the usual `Diag`/`DiagCtxt` level, so we must augment `bug`
                // in a lower-level fashion.
                bug.arg("level", bug.level);
                let msg = crate::fluent_generated::errors_invalid_flushed_delayed_diagnostic_level;
                let msg = self.eagerly_translate_for_subdiag(&bug, msg); // after the `arg` call
                bug.sub(Note, msg, bug.span.primary_span().unwrap().into());
            }
            bug.level = Bug;

            self.emit_diagnostic(bug, None);
        }

        // Panic with `DelayedBugPanic` to avoid "unexpected panic" messages.
        panic::panic_any(DelayedBugPanic);
    }

    fn panic_if_treat_err_as_bug(&self) {
        if self.treat_err_as_bug() {
            let n = self.flags.treat_err_as_bug.map(|c| c.get()).unwrap();
            assert_eq!(n, self.err_guars.len() + self.lint_err_guars.len());
            if n == 1 {
                panic!("aborting due to `-Z treat-err-as-bug=1`");
            } else {
                panic!("aborting after {n} errors due to `-Z treat-err-as-bug={n}`");
            }
        }
    }
}

struct DelayedDiagInner {
    inner: DiagInner,
    note: Backtrace,
}

impl DelayedDiagInner {
    fn with_backtrace(diagnostic: DiagInner, backtrace: Backtrace) -> Self {
        DelayedDiagInner { inner: diagnostic, note: backtrace }
    }

    fn decorate(self, dcx: &DiagCtxtInner) -> DiagInner {
        // We are at the `DiagInner`/`DiagCtxtInner` level rather than the
        // usual `Diag`/`DiagCtxt` level, so we must construct `diag` in a
        // lower-level fashion.
        let mut diag = self.inner;
        let msg = match self.note.status() {
            BacktraceStatus::Captured => crate::fluent_generated::errors_delayed_at_with_newline,
            // Avoid the needless newline when no backtrace has been captured,
            // the display impl should just be a single line.
            _ => crate::fluent_generated::errors_delayed_at_without_newline,
        };
        diag.arg("emitted_at", diag.emitted_at.clone());
        diag.arg("note", self.note);
        let msg = dcx.eagerly_translate_for_subdiag(&diag, msg); // after the `arg` calls
        diag.sub(Note, msg, diag.span.primary_span().unwrap_or(DUMMY_SP).into());
        diag
    }
}

/// | Level        | is_error | EmissionGuarantee            | Top-level | Sub | Used in lints?
/// | -----        | -------- | -----------------            | --------- | --- | --------------
/// | Bug          | yes      | BugAbort                     | yes       | -   | -
/// | Fatal        | yes      | FatalAbort/FatalError[^star] | yes       | -   | -
/// | Error        | yes      | ErrorGuaranteed              | yes       | -   | yes
/// | DelayedBug   | yes      | ErrorGuaranteed              | yes       | -   | -
/// | ForceWarning | -        | ()                           | yes       | -   | lint-only
/// | Warning      | -        | ()                           | yes       | yes | yes
/// | Note         | -        | ()                           | rare      | yes | -
/// | OnceNote     | -        | ()                           | -         | yes | lint-only
/// | Help         | -        | ()                           | rare      | yes | -
/// | OnceHelp     | -        | ()                           | -         | yes | lint-only
/// | FailureNote  | -        | ()                           | rare      | -   | -
/// | Allow        | -        | ()                           | yes       | -   | lint-only
/// | Expect       | -        | ()                           | yes       | -   | lint-only
///
/// [^star]: `FatalAbort` normally, `FatalError` in the non-aborting "almost fatal" case that is
///     occasionally used.
///
#[derive(Copy, PartialEq, Eq, Clone, Hash, Debug, Encodable, Decodable)]
pub enum Level {
    /// For bugs in the compiler. Manifests as an ICE (internal compiler error) panic.
    Bug,

    /// An error that causes an immediate abort. Used for things like configuration errors,
    /// internal overflows, some file operation errors.
    Fatal,

    /// An error in the code being compiled, which prevents compilation from finishing. This is the
    /// most common case.
    Error,

    /// This is a strange one: lets you register an error without emitting it. If compilation ends
    /// without any other errors occurring, this will be emitted as a bug. Otherwise, it will be
    /// silently dropped. I.e. "expect other errors are emitted" semantics. Useful on code paths
    /// that should only be reached when compiling erroneous code.
    DelayedBug,

    /// A `force-warn` lint warning about the code being compiled. Does not prevent compilation
    /// from finishing.
    ///
    /// Requires a [`LintExpectationId`] for expected lint diagnostics. In all other cases this
    /// should be `None`.
    ForceWarning,

    /// A warning about the code being compiled. Does not prevent compilation from finishing.
    /// Will be skipped if `can_emit_warnings` is false.
    Warning,

    /// A message giving additional context.
    Note,

    /// A note that is only emitted once.
    OnceNote,

    /// A message suggesting how to fix something.
    Help,

    /// A help that is only emitted once.
    OnceHelp,

    /// Similar to `Note`, but used in cases where compilation has failed. When printed for human
    /// consumption, it doesn't have any kind of `note:` label.
    FailureNote,

    /// Only used for lints.
    Allow,

    /// Only used for lints. Requires a [`LintExpectationId`] for silencing the lints.
    Expect,
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
            Bug | Fatal | Error | DelayedBug => {
                spec.set_fg(Some(Color::Red)).set_intense(true);
            }
            ForceWarning | Warning => {
                spec.set_fg(Some(Color::Yellow)).set_intense(cfg!(windows));
            }
            Note | OnceNote => {
                spec.set_fg(Some(Color::Green)).set_intense(true);
            }
            Help | OnceHelp => {
                spec.set_fg(Some(Color::Cyan)).set_intense(true);
            }
            FailureNote => {}
            Allow | Expect => unreachable!(),
        }
        spec
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Bug | DelayedBug => "error: internal compiler error",
            Fatal | Error => "error",
            ForceWarning | Warning => "warning",
            Note | OnceNote => "note",
            Help | OnceHelp => "help",
            FailureNote => "failure-note",
            Allow | Expect => unreachable!(),
        }
    }

    pub fn is_failure_note(&self) -> bool {
        matches!(*self, FailureNote)
    }

    // Can this level be used in a subdiagnostic message?
    fn can_be_subdiag(&self) -> bool {
        match self {
            Bug | DelayedBug | Fatal | Error | ForceWarning | FailureNote | Allow | Expect => false,

            Warning | Note | Help | OnceNote | OnceHelp => true,
        }
    }
}

// FIXME(eddyb) this doesn't belong here AFAICT, should be moved to callsite.
pub fn elided_lifetime_in_path_suggestion(
    source_map: &SourceMap,
    n: usize,
    path_span: Span,
    incl_angl_brckt: bool,
    insertion_span: Span,
) -> ElidedLifetimeInPathSubdiag {
    let expected = ExpectedLifetimeParameter { span: path_span, count: n };
    // Do not try to suggest anything if generated by a proc-macro.
    let indicate = source_map.is_span_accessible(insertion_span).then(|| {
        let anon_lts = vec!["'_"; n].join(", ");
        let suggestion =
            if incl_angl_brckt { format!("<{anon_lts}>") } else { format!("{anon_lts}, ") };

        IndicateAnonymousLifetime { span: insertion_span.shrink_to_hi(), count: n, suggestion }
    });

    ElidedLifetimeInPathSubdiag { expected, indicate }
}

pub fn report_ambiguity_error<'a, G: EmissionGuarantee>(
    diag: &mut Diag<'a, G>,
    ambiguity: rustc_lint_defs::AmbiguityErrorDiag,
) {
    diag.span_label(ambiguity.label_span, ambiguity.label_msg);
    diag.note(ambiguity.note_msg);
    diag.span_note(ambiguity.b1_span, ambiguity.b1_note_msg);
    for help_msg in ambiguity.b1_help_msgs {
        diag.help(help_msg);
    }
    diag.span_note(ambiguity.b2_span, ambiguity.b2_note_msg);
    for help_msg in ambiguity.b2_help_msgs {
        diag.help(help_msg);
    }
}

/// Grammatical tool for displaying messages to end users in a nice form.
///
/// Returns "an" if the given string starts with a vowel, and "a" otherwise.
pub fn a_or_an(s: &str) -> &'static str {
    let mut chars = s.chars();
    let Some(mut first_alpha_char) = chars.next() else {
        return "a";
    };
    if first_alpha_char == '`' {
        let Some(next) = chars.next() else {
            return "a";
        };
        first_alpha_char = next;
    }
    if ["a", "e", "i", "o", "u", "&"].contains(&&first_alpha_char.to_lowercase().to_string()[..]) {
        "an"
    } else {
        "a"
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum TerminalUrl {
    No,
    Yes,
    Auto,
}
