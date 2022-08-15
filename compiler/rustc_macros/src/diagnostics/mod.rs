mod diagnostic;
mod diagnostic_builder;
mod error;
mod fluent;
mod subdiagnostic;
mod utils;

use diagnostic::{LintDiagnosticDerive, SessionDiagnosticDerive};
pub(crate) use fluent::fluent_messages;
use proc_macro2::TokenStream;
use quote::format_ident;
use subdiagnostic::SessionSubdiagnosticDerive;
use synstructure::Structure;

/// Implements `#[derive(SessionDiagnostic)]`, which allows for errors to be specified as a struct,
/// independent from the actual diagnostics emitting code.
///
/// ```ignore (rust)
/// # extern crate rustc_errors;
/// # use rustc_errors::Applicability;
/// # extern crate rustc_span;
/// # use rustc_span::{symbol::Ident, Span};
/// # extern crate rust_middle;
/// # use rustc_middle::ty::Ty;
/// #[derive(SessionDiagnostic)]
/// #[error(borrowck::move_out_of_borrow, code = "E0505")]
/// pub struct MoveOutOfBorrowError<'tcx> {
///     pub name: Ident,
///     pub ty: Ty<'tcx>,
///     #[primary_span]
///     #[label]
///     pub span: Span,
///     #[label(borrowck::first_borrow_label)]
///     pub first_borrow_span: Span,
///     #[suggestion(code = "{name}.clone()")]
///     pub clone_sugg: Option<(Span, Applicability)>
/// }
/// ```
///
/// ```fluent
/// move_out_of_borrow = cannot move out of {$name} because it is borrowed
///     .label = cannot move out of borrow
///     .first_borrow_label = `{$ty}` first borrowed here
///     .suggestion = consider cloning here
/// ```
///
/// Then, later, to emit the error:
///
/// ```ignore (rust)
/// sess.emit_err(MoveOutOfBorrowError {
///     expected,
///     actual,
///     span,
///     first_borrow_span,
///     clone_sugg: Some(suggestion, Applicability::MachineApplicable),
/// });
/// ```
///
/// See rustc dev guide for more examples on using the `#[derive(SessionDiagnostic)]`:
/// <https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-structs.html>
pub fn session_diagnostic_derive(s: Structure<'_>) -> TokenStream {
    SessionDiagnosticDerive::new(format_ident!("diag"), format_ident!("sess"), s).into_tokens()
}

/// Implements `#[derive(LintDiagnostic)]`, which allows for lints to be specified as a struct,
/// independent from the actual lint emitting code.
///
/// ```ignore (rust)
/// #[derive(LintDiagnostic)]
/// #[lint(lint::atomic_ordering_invalid_fail_success)]
/// pub struct AtomicOrderingInvalidLint {
///     method: Symbol,
///     success_ordering: Symbol,
///     fail_ordering: Symbol,
///     #[label(lint::fail_label)]
///     fail_order_arg_span: Span,
///     #[label(lint::success_label)]
///     #[suggestion(
///         code = "std::sync::atomic::Ordering::{success_suggestion}",
///         applicability = "maybe-incorrect"
///     )]
///     success_order_arg_span: Span,
/// }
/// ```
///
/// ```fluent
/// lint_atomic_ordering_invalid_fail_success = `{$method}`'s success ordering must be at least as strong as its failure ordering
///     .fail_label = `{$fail_ordering}` failure ordering
///     .success_label = `{$success_ordering}` success ordering
///     .suggestion = consider using `{$success_suggestion}` success ordering instead
/// ```
///
/// Then, later, to emit the error:
///
/// ```ignore (rust)
/// cx.struct_span_lint(INVALID_ATOMIC_ORDERING, fail_order_arg_span, AtomicOrderingInvalidLint {
///     method,
///     success_ordering,
///     fail_ordering,
///     fail_order_arg_span,
///     success_order_arg_span,
/// });
/// ```
///
/// See rustc dev guide for more examples on using the `#[derive(LintDiagnostic)]`:
/// <https://rustc-dev-guide.rust-lang.org/diagnostics/sessiondiagnostic.html>
pub fn lint_diagnostic_derive(s: Structure<'_>) -> TokenStream {
    LintDiagnosticDerive::new(format_ident!("diag"), s).into_tokens()
}

/// Implements `#[derive(SessionSubdiagnostic)]`, which allows for labels, notes, helps and
/// suggestions to be specified as a structs or enums, independent from the actual diagnostics
/// emitting code or diagnostic derives.
///
/// ```ignore (rust)
/// #[derive(SessionSubdiagnostic)]
/// pub enum ExpectedIdentifierLabel<'tcx> {
///     #[label(parser::expected_identifier)]
///     WithoutFound {
///         #[primary_span]
///         span: Span,
///     }
///     #[label(parser::expected_identifier_found)]
///     WithFound {
///         #[primary_span]
///         span: Span,
///         found: String,
///     }
/// }
///
/// #[derive(SessionSubdiagnostic)]
/// #[suggestion_verbose(parser::raw_identifier)]
/// pub struct RawIdentifierSuggestion<'tcx> {
///     #[primary_span]
///     span: Span,
///     #[applicability]
///     applicability: Applicability,
///     ident: Ident,
/// }
/// ```
///
/// ```fluent
/// parser_expected_identifier = expected identifier
///
/// parser_expected_identifier-found = expected identifier, found {$found}
///
/// parser_raw_identifier = escape `{$ident}` to use it as an identifier
/// ```
///
/// Then, later, to add the subdiagnostic:
///
/// ```ignore (rust)
/// diag.subdiagnostic(ExpectedIdentifierLabel::WithoutFound { span });
///
/// diag.subdiagnostic(RawIdentifierSuggestion { span, applicability, ident });
/// ```
pub fn session_subdiagnostic_derive(s: Structure<'_>) -> TokenStream {
    SessionSubdiagnosticDerive::new(s).into_tokens()
}
