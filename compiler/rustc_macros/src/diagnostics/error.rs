use proc_macro::{Diagnostic, Level, MultiSpan};
use proc_macro2::TokenStream;
use quote::quote;
use syn;

#[derive(Debug)]
pub(crate) enum SessionDiagnosticDeriveError {
    SynError(syn::Error),
    ErrorHandled,
}

impl SessionDiagnosticDeriveError {
    pub(crate) fn to_compile_error(self) -> TokenStream {
        match self {
            SessionDiagnosticDeriveError::SynError(e) => e.to_compile_error(),
            SessionDiagnosticDeriveError::ErrorHandled => {
                // Return ! to avoid having to create a blank DiagnosticBuilder to return when an
                // error has already been emitted to the compiler.
                quote! {
                    { unreachable!(); }
                }
            }
        }
    }
}

pub(crate) fn span_err(span: impl MultiSpan, msg: &str) -> Diagnostic {
    Diagnostic::spanned(span, Level::Error, msg)
}

/// For methods that return a `Result<_, SessionDiagnosticDeriveError>`:
///
/// Emit a diagnostic on span `$span` with msg `$msg` (optionally performing additional decoration
/// using the `FnOnce` passed in `diag`) and return `Err(ErrorHandled)`.
macro_rules! throw_span_err {
    ($span:expr, $msg:expr) => {{ throw_span_err!($span, $msg, |diag| diag) }};
    ($span:expr, $msg:expr, $f:expr) => {{
        return Err(crate::diagnostics::error::_throw_span_err($span, $msg, $f));
    }};
}

pub(crate) use throw_span_err;

/// When possible, prefer using `throw_span_err!` over using this function directly. This only
/// exists as a function to constrain `f` to an `impl FnOnce`.
pub(crate) fn _throw_span_err(
    span: impl MultiSpan,
    msg: &str,
    f: impl FnOnce(Diagnostic) -> Diagnostic,
) -> SessionDiagnosticDeriveError {
    let diag = span_err(span, msg);
    f(diag).emit();
    SessionDiagnosticDeriveError::ErrorHandled
}

impl From<syn::Error> for SessionDiagnosticDeriveError {
    fn from(e: syn::Error) -> Self {
        SessionDiagnosticDeriveError::SynError(e)
    }
}
