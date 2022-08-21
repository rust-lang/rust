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

#[derive(SessionDiagnostic)]
#[error(symbol_mangling::invalid_trait_item)]
pub struct InvalidTraitItem<'a> {
    #[primary_span]
    pub span: Span,
    pub demangling_formatted: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(symbol_mangling::alt_invalid_trait_item)]
pub struct AltInvalidTraitItem<'a> {
    #[primary_span]
    pub span: Span,
    pub alt_demangling_formatted: &'a str,
}
