mod diagnostic;
mod error;
mod subdiagnostic;
mod utils;

use diagnostic::SessionDiagnosticDerive;
use proc_macro2::TokenStream;
use quote::format_ident;
use subdiagnostic::SessionSubdiagnosticDerive;
use synstructure::Structure;

/// Implements `#[derive(SessionDiagnostic)]`, which allows for errors to be specified as a struct,
/// independent from the actual diagnostics emitting code.
///
/// ```ignore (pseudo-rust)
/// # extern crate rustc_errors;
/// # use rustc_errors::Applicability;
/// # extern crate rustc_span;
/// # use rustc_span::{symbol::Ident, Span};
/// # extern crate rust_middle;
/// # use rustc_middle::ty::Ty;
/// #[derive(SessionDiagnostic)]
/// #[error(code = "E0505", slug = "borrowck-move-out-of-borrow")]
/// pub struct MoveOutOfBorrowError<'tcx> {
///     pub name: Ident,
///     pub ty: Ty<'tcx>,
///     #[primary_span]
///     #[label]
///     pub span: Span,
///     #[label = "first-borrow-label"]
///     pub first_borrow_span: Span,
///     #[suggestion(code = "{name}.clone()")]
///     pub clone_sugg: Option<(Span, Applicability)>
/// }
/// ```
///
/// ```fluent
/// move-out-of-borrow = cannot move out of {$name} because it is borrowed
///     .label = cannot move out of borrow
///     .first-borrow-label = `{$ty}` first borrowed here
///     .suggestion = consider cloning here
/// ```
///
/// Then, later, to emit the error:
///
/// ```ignore (pseudo-rust)
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
/// <https://rustc-dev-guide.rust-lang.org/diagnostics/sessiondiagnostic.html>
pub fn session_diagnostic_derive(s: Structure<'_>) -> TokenStream {
    // Names for the diagnostic we build and the session we build it from.
    let diag = format_ident!("diag");
    let sess = format_ident!("sess");

    SessionDiagnosticDerive::new(diag, sess, s).into_tokens()
}

/// Implements `#[derive(SessionSubdiagnostic)]`, which allows for labels, notes, helps and
/// suggestions to be specified as a structs or enums, independent from the actual diagnostics
/// emitting code or diagnostic derives.
///
/// ```ignore (pseudo-rust)
/// #[derive(SessionSubdiagnostic)]
/// pub enum ExpectedIdentifierLabel<'tcx> {
///     #[label(slug = "parser-expected-identifier")]
///     WithoutFound {
///         #[primary_span]
///         span: Span,
///     }
///     #[label(slug = "parser-expected-identifier-found")]
///     WithFound {
///         #[primary_span]
///         span: Span,
///         found: String,
///     }
/// }
///
/// #[derive(SessionSubdiagnostic)]
/// #[suggestion_verbose(slug = "parser-raw-identifier")]
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
/// parser-expected-identifier = expected identifier
///
/// parser-expected-identifier-found = expected identifier, found {$found}
///
/// parser-raw-identifier = escape `{$ident}` to use it as an identifier
/// ```
///
/// Then, later, to add the subdiagnostic:
///
/// ```ignore (pseudo-rust)
/// diag.subdiagnostic(ExpectedIdentifierLabel::WithoutFound { span });
///
/// diag.subdiagnostic(RawIdentifierSuggestion { span, applicability, ident });
/// ```
pub fn session_subdiagnostic_derive(s: Structure<'_>) -> TokenStream {
    SessionSubdiagnosticDerive::new(s).into_tokens()
}
