use proc_macro::{Diagnostic, Level, MultiSpan};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, Attribute, Error as SynError, Meta, NestedMeta};

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
                // Return ! to avoid having to create a blank DiagnosticBuilder to return when an
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
pub(crate) fn span_err(span: impl MultiSpan, msg: &str) -> Diagnostic {
    Diagnostic::spanned(span, Level::Error, msg)
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
pub(crate) fn invalid_attr(attr: &Attribute, meta: &Meta) -> Diagnostic {
    let span = attr.span().unwrap();
    let path = path_to_string(&attr.path);
    match meta {
        Meta::Path(_) => span_err(span, &format!("`#[{}]` is not a valid attribute", path)),
        Meta::NameValue(_) => {
            span_err(span, &format!("`#[{} = ...]` is not a valid attribute", path))
        }
        Meta::List(_) => span_err(span, &format!("`#[{}(...)]` is not a valid attribute", path)),
    }
}

/// Emit a error diagnostic for an invalid attribute (optionally performing additional decoration
/// using the `FnOnce` passed in `diag`) and return `Err(ErrorHandled)`.
///
/// For methods that return a `Result<_, DiagnosticDeriveError>`:
macro_rules! throw_invalid_attr {
    ($attr:expr, $meta:expr) => {{ throw_invalid_attr!($attr, $meta, |diag| diag) }};
    ($attr:expr, $meta:expr, $f:expr) => {{
        let diag = crate::diagnostics::error::invalid_attr($attr, $meta);
        return Err(crate::diagnostics::error::_throw_err(diag, $f));
    }};
}

pub(crate) use throw_invalid_attr;

/// Returns an error diagnostic for an invalid nested attribute.
pub(crate) fn invalid_nested_attr(attr: &Attribute, nested: &NestedMeta) -> Diagnostic {
    let name = attr.path.segments.last().unwrap().ident.to_string();
    let name = name.as_str();

    let span = nested.span().unwrap();
    let meta = match nested {
        syn::NestedMeta::Meta(meta) => meta,
        syn::NestedMeta::Lit(_) => {
            return span_err(span, &format!("`#[{}(\"...\")]` is not a valid attribute", name));
        }
    };

    let span = meta.span().unwrap();
    let path = path_to_string(meta.path());
    match meta {
        Meta::NameValue(..) => {
            span_err(span, &format!("`#[{}({} = ...)]` is not a valid attribute", name, path))
        }
        Meta::Path(..) => {
            span_err(span, &format!("`#[{}({})]` is not a valid attribute", name, path))
        }
        Meta::List(..) => {
            span_err(span, &format!("`#[{}({}(...))]` is not a valid attribute", name, path))
        }
    }
}

/// Emit a error diagnostic for an invalid nested attribute (optionally performing additional
/// decoration using the `FnOnce` passed in `diag`) and return `Err(ErrorHandled)`.
///
/// For methods that return a `Result<_, DiagnosticDeriveError>`:
macro_rules! throw_invalid_nested_attr {
    ($attr:expr, $nested_attr:expr) => {{ throw_invalid_nested_attr!($attr, $nested_attr, |diag| diag) }};
    ($attr:expr, $nested_attr:expr, $f:expr) => {{
        let diag = crate::diagnostics::error::invalid_nested_attr($attr, $nested_attr);
        return Err(crate::diagnostics::error::_throw_err(diag, $f));
    }};
}

pub(crate) use throw_invalid_nested_attr;
