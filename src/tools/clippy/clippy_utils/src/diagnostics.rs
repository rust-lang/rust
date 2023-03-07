//! Clippy wrappers around rustc's diagnostic functions.
//!
//! These functions are used by the `INTERNAL_METADATA_COLLECTOR` lint to collect the corresponding
//! lint applicability. Please make sure that you update the `LINT_EMISSION_FUNCTIONS` variable in
//! `clippy_lints::utils::internal_lints::metadata_collector` when a new function is added
//! or renamed.
//!
//! Thank you!
//! ~The `INTERNAL_METADATA_COLLECTOR` lint

use rustc_errors::{Applicability, Diagnostic, MultiSpan};
use rustc_hir::HirId;
use rustc_lint::{LateContext, Lint, LintContext};
use rustc_span::source_map::Span;
use std::env;

fn docs_link(diag: &mut Diagnostic, lint: &'static Lint) {
    if env::var("CLIPPY_DISABLE_DOCS_LINKS").is_err() {
        if let Some(lint) = lint.name_lower().strip_prefix("clippy::") {
            diag.help(format!(
                "for further information visit https://rust-lang.github.io/rust-clippy/{}/index.html#{lint}",
                &option_env!("RUST_RELEASE_NUM").map_or("master".to_string(), |n| {
                    // extract just major + minor version and ignore patch versions
                    format!("rust-{}", n.rsplit_once('.').unwrap().1)
                })
            ));
        }
    }
}

/// Emit a basic lint message with a `msg` and a `span`.
///
/// This is the most primitive of our lint emission methods and can
/// be a good way to get a new lint started.
///
/// Usually it's nicer to provide more context for lint messages.
/// Be sure the output is understandable when you use this method.
///
/// # Example
///
/// ```ignore
/// error: usage of mem::forget on Drop type
///   --> $DIR/mem_forget.rs:17:5
///    |
/// 17 |     std::mem::forget(seven);
///    |     ^^^^^^^^^^^^^^^^^^^^^^^
/// ```
pub fn span_lint<T: LintContext>(cx: &T, lint: &'static Lint, sp: impl Into<MultiSpan>, msg: &str) {
    cx.struct_span_lint(lint, sp, msg, |diag| {
        docs_link(diag, lint);
        diag
    });
}

/// Same as `span_lint` but with an extra `help` message.
///
/// Use this if you want to provide some general help but
/// can't provide a specific machine applicable suggestion.
///
/// The `help` message can be optionally attached to a `Span`.
///
/// If you change the signature, remember to update the internal lint `CollapsibleCalls`
///
/// # Example
///
/// ```text
/// error: constant division of 0.0 with 0.0 will always result in NaN
///   --> $DIR/zero_div_zero.rs:6:25
///    |
/// 6  |     let other_f64_nan = 0.0f64 / 0.0;
///    |                         ^^^^^^^^^^^^
///    |
///    = help: consider using `f64::NAN` if you would like a constant representing NaN
/// ```
pub fn span_lint_and_help<T: LintContext>(
    cx: &T,
    lint: &'static Lint,
    span: impl Into<MultiSpan>,
    msg: &str,
    help_span: Option<Span>,
    help: &str,
) {
    cx.struct_span_lint(lint, span, msg, |diag| {
        if let Some(help_span) = help_span {
            diag.span_help(help_span, help);
        } else {
            diag.help(help);
        }
        docs_link(diag, lint);
        diag
    });
}

/// Like `span_lint` but with a `note` section instead of a `help` message.
///
/// The `note` message is presented separately from the main lint message
/// and is attached to a specific span:
///
/// If you change the signature, remember to update the internal lint `CollapsibleCalls`
///
/// # Example
///
/// ```text
/// error: calls to `std::mem::forget` with a reference instead of an owned value. Forgetting a reference does nothing.
///   --> $DIR/drop_forget_ref.rs:10:5
///    |
/// 10 |     forget(&SomeStruct);
///    |     ^^^^^^^^^^^^^^^^^^^
///    |
///    = note: `-D clippy::forget-ref` implied by `-D warnings`
/// note: argument has type &SomeStruct
///   --> $DIR/drop_forget_ref.rs:10:12
///    |
/// 10 |     forget(&SomeStruct);
///    |            ^^^^^^^^^^^
/// ```
pub fn span_lint_and_note<T: LintContext>(
    cx: &T,
    lint: &'static Lint,
    span: impl Into<MultiSpan>,
    msg: &str,
    note_span: Option<Span>,
    note: &str,
) {
    cx.struct_span_lint(lint, span, msg, |diag| {
        if let Some(note_span) = note_span {
            diag.span_note(note_span, note);
        } else {
            diag.note(note);
        }
        docs_link(diag, lint);
        diag
    });
}

/// Like `span_lint` but allows to add notes, help and suggestions using a closure.
///
/// If you need to customize your lint output a lot, use this function.
/// If you change the signature, remember to update the internal lint `CollapsibleCalls`
pub fn span_lint_and_then<C, S, F>(cx: &C, lint: &'static Lint, sp: S, msg: &str, f: F)
where
    C: LintContext,
    S: Into<MultiSpan>,
    F: FnOnce(&mut Diagnostic),
{
    cx.struct_span_lint(lint, sp, msg, |diag| {
        f(diag);
        docs_link(diag, lint);
        diag
    });
}

pub fn span_lint_hir(cx: &LateContext<'_>, lint: &'static Lint, hir_id: HirId, sp: Span, msg: &str) {
    cx.tcx.struct_span_lint_hir(lint, hir_id, sp, msg, |diag| {
        docs_link(diag, lint);
        diag
    });
}

pub fn span_lint_hir_and_then(
    cx: &LateContext<'_>,
    lint: &'static Lint,
    hir_id: HirId,
    sp: impl Into<MultiSpan>,
    msg: &str,
    f: impl FnOnce(&mut Diagnostic),
) {
    cx.tcx.struct_span_lint_hir(lint, hir_id, sp, msg, |diag| {
        f(diag);
        docs_link(diag, lint);
        diag
    });
}

/// Add a span lint with a suggestion on how to fix it.
///
/// These suggestions can be parsed by rustfix to allow it to automatically fix your code.
/// In the example below, `help` is `"try"` and `sugg` is the suggested replacement `".any(|x| x >
/// 2)"`.
///
/// If you change the signature, remember to update the internal lint `CollapsibleCalls`
///
/// # Example
///
/// ```text
/// error: This `.fold` can be more succinctly expressed as `.any`
/// --> $DIR/methods.rs:390:13
///     |
/// 390 |     let _ = (0..3).fold(false, |acc, x| acc || x > 2);
///     |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `.any(|x| x > 2)`
///     |
///     = note: `-D fold-any` implied by `-D warnings`
/// ```
#[cfg_attr(feature = "internal", allow(clippy::collapsible_span_lint_calls))]
pub fn span_lint_and_sugg<T: LintContext>(
    cx: &T,
    lint: &'static Lint,
    sp: Span,
    msg: &str,
    help: &str,
    sugg: String,
    applicability: Applicability,
) {
    span_lint_and_then(cx, lint, sp, msg, |diag| {
        diag.span_suggestion(sp, help, sugg, applicability);
    });
}

/// Create a suggestion made from several `span → replacement`.
///
/// Note: in the JSON format (used by `compiletest_rs`), the help message will
/// appear once per
/// replacement. In human-readable format though, it only appears once before
/// the whole suggestion.
pub fn multispan_sugg<I>(diag: &mut Diagnostic, help_msg: &str, sugg: I)
where
    I: IntoIterator<Item = (Span, String)>,
{
    multispan_sugg_with_applicability(diag, help_msg, Applicability::Unspecified, sugg);
}

/// Create a suggestion made from several `span → replacement`.
///
/// rustfix currently doesn't support the automatic application of suggestions with
/// multiple spans. This is tracked in issue [rustfix#141](https://github.com/rust-lang/rustfix/issues/141).
/// Suggestions with multiple spans will be silently ignored.
pub fn multispan_sugg_with_applicability<I>(
    diag: &mut Diagnostic,
    help_msg: &str,
    applicability: Applicability,
    sugg: I,
) where
    I: IntoIterator<Item = (Span, String)>,
{
    diag.multipart_suggestion(help_msg, sugg.into_iter().collect(), applicability);
}
