//! Errors emitted by symbol_mangling.

use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[error(symbol_mangling::invalid_symbol_name)]
pub struct InvalidSymbolName<'a> {
    #[primary_span]
    pub span: Span,
    pub mangled_formatted: &'a str,
}
