//! Errors emitted by symbol_mangling.

use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(symbol_mangling::invalid_symbol_name)]
pub struct InvalidSymbolName {
    #[primary_span]
    pub span: Span,
    pub mangled_formatted: String,
}

#[derive(SessionDiagnostic)]
#[diag(symbol_mangling::invalid_trait_item)]
pub struct InvalidTraitItem {
    #[primary_span]
    pub span: Span,
    pub demangling_formatted: String,
}

#[derive(SessionDiagnostic)]
#[diag(symbol_mangling::alt_invalid_trait_item)]
pub struct AltInvalidTraitItem {
    #[primary_span]
    pub span: Span,
    pub alt_demangling_formatted: String,
}

#[derive(SessionDiagnostic)]
#[diag(symbol_mangling::invalid_def_path)]
pub struct InvalidDefPath {
    #[primary_span]
    pub span: Span,
    pub def_path: String,
}
