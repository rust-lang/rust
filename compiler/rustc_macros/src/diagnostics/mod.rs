mod diagnostic;
mod diagnostic_builder;
mod error;
mod message;
mod msg_macro;
mod subdiagnostic;
mod utils;

use diagnostic::DiagnosticDerive;
pub(super) use msg_macro::msg_macro;
use proc_macro2::TokenStream;
use subdiagnostic::SubdiagnosticDerive;
use synstructure::Structure;

/// Implements `#[derive(Diagnostic)]`, which allows for errors to be specified as a struct,
/// independent from the actual diagnostics emitting code.
///
/// ```ignore (rust)
/// # extern crate rustc_errors;
/// # use rustc_errors::Applicability;
/// # extern crate rustc_span;
/// # use rustc_span::{Ident, Span};
/// # extern crate rust_middle;
/// # use rustc_middle::ty::Ty;
/// #[derive(Diagnostic)]
/// #[diag("this is an example message", code = E0123)]
/// pub(crate) struct ExampleError<'tcx> {
///     pub name: Ident,
///     pub ty: Ty<'tcx>,
///     #[primary_span]
///     #[label("with a label")]
///     pub span: Span,
///     #[label("with a label")]
///     pub other_span: Span,
///     #[suggestion("with a suggestion", code = "{name}.clone()")]
///     pub opt_sugg: Option<(Span, Applicability)>,
/// }
/// ```
///
/// Then, later, to emit the error:
///
/// ```ignore (rust)
/// sess.emit_err(ExampleError {
///     name, ty, span, other_span, opt_sugg
/// });
/// ```
///
/// See rustc dev guide for more examples on using the `#[derive(Diagnostic)]`:
/// <https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-structs.html>
pub(super) fn diagnostic_derive(s: Structure<'_>) -> TokenStream {
    DiagnosticDerive::new(s).into_tokens()
}

/// Implements `#[derive(Subdiagnostic)]`, which allows for labels, notes, helps and
/// suggestions to be specified as a structs or enums, independent from the actual diagnostics
/// emitting code or diagnostic derives.
///
/// ```ignore (rust)
/// #[derive(Subdiagnostic)]
/// pub(crate) enum BuiltinUnusedDocCommentSub {
///     #[help("use `//` for a plain comment")]
///     PlainHelp,
///     #[help("use `/* */` for a plain comment")]
///     BlockHelp,
/// }
/// ```
/// Then, later, use the subdiagnostic in a diagnostic:
///
/// ```ignore (rust)
/// #[derive(Diagnostic)]
/// #[diag("unused doc comment")]
/// pub(crate) struct BuiltinUnusedDocComment<'a> {
///     pub kind: &'a str,
///     #[label("rustdoc does not generate documentation for {$kind}")]
///     pub label: Span,
///     #[subdiagnostic]
///     pub sub: BuiltinUnusedDocCommentSub,
/// }
/// ```
pub(super) fn subdiagnostic_derive(s: Structure<'_>) -> TokenStream {
    SubdiagnosticDerive::new().into_tokens(s)
}
