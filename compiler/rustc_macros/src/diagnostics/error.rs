use proc_macro::{Diagnostic, Level, MultiSpan};
use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Attribute, Error as SynError, Meta};

#[derive(Debug)]
pub(crate) enum DiagnosticDeriveError {
    SynError(SynError),
    ErrorHandled,
}

impl DiagnosticDeriveError {
    pub(crate) fn to_compile_error(self) -> TokenStream {
        match self {
            DiagnosticDeriveError::SynError(e) => e.to_compile_error(),
            DiagnosticDeriveError::ErrorHandled => {
                // Return ! to avoid having to create a blank Diag to return when an
                // error has already been emitted to the compiler.
                quote! {
                    { unreachable!(); }
                }
            }
        }
    }
}

impl From<SynError> for DiagnosticDeriveError {
    fn from(e: SynError) -> Self {
        DiagnosticDeriveError::SynError(e)
    }
}

/// Helper function for use with `throw_*` macros - constraints `$f` to an `impl FnOnce`.
pub(crate) fn _throw_err(
    diag: Diagnostic,
    f: impl FnOnce(Diagnostic) -> Diagnostic,
) -> DiagnosticDeriveError {
    f(diag).emit();
    DiagnosticDeriveError::ErrorHandled
}

/// Helper function for printing `syn::Path` - doesn't handle arguments in paths and these are
/// unlikely to come up much in use of the macro.
fn path_to_string(path: &syn::Path) -> String {
    let mut out = String::new();
    for (i, segment) in path.segments.iter().enumerate() {
        if i > 0 || path.leading_colon.is_some() {
            out.push_str("::");
        }
        out.push_str(&segment.ident.to_string());
    }
    out
}

/// Returns an error diagnostic on span `span` with msg `msg`.
#[must_use]
pub(crate) fn span_err<T: Into<String>>(span: impl MultiSpan, msg: T) -> Diagnostic {
    Diagnostic::spanned(span, Level::Error, format!("derive(Diagnostic): {}", msg.into()))
}

/// Emit a diagnostic on span `$span` with msg `$msg` (optionally performing additional decoration
/// using the `FnOnce` passed in `diag`) and return `Err(ErrorHandled)`.
///
/// For methods that return a `Result<_, DiagnosticDeriveError>`:
macro_rules! throw_span_err {
    ($span:expr, $msg:expr) => {{ throw_span_err!($span, $msg, |diag| diag) }};
    ($span:expr, $msg:expr, $f:expr) => {{
        let diag = span_err($span, $msg);
        return Err(crate::diagnostics::error::_throw_err(diag, $f));
    }};
}

pub(crate) use throw_span_err;

/// Returns an error diagnostic for an invalid attribute.
pub(crate) fn invalid_attr(attr: &Attribute) -> Diagnostic {
    let span = attr.span().unwrap();
    let path = path_to_string(attr.path());
    match attr.meta {
        Meta::Path(_) => span_err(span, format!("`#[{path}]` is not a valid attribute")),
        Meta::NameValue(_) => span_err(span, format!("`#[{path} = ...]` is not a valid attribute")),
        Meta::List(_) => span_err(span, format!("`#[{path}(...)]` is not a valid attribute")),
    }
}

/// Emit an error diagnostic for an invalid attribute (optionally performing additional decoration
/// using the `FnOnce` passed in `diag`) and return `Err(ErrorHandled)`.
///
/// For methods that return a `Result<_, DiagnosticDeriveError>`:
macro_rules! throw_invalid_attr {
    ($attr:expr) => {{ throw_invalid_attr!($attr, |diag| diag) }};
    ($attr:expr, $f:expr) => {{
        let diag = crate::diagnostics::error::invalid_attr($attr);
        return Err(crate::diagnostics::error::_throw_err(diag, $f));
    }};
}

pub(crate) use throw_invalid_attr;
