use rustc_hir::Target;
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(resolve_lang_item_on_incorrect_target, code = "E0718")]
pub struct LangItemOnIncorrectTarget {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
    pub expected_target: Target,
    pub actual_target: Target,
}

#[derive(Diagnostic)]
#[diag(resolve_incorrect_target, code = "E0718")]
pub struct IncorrectTarget<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub generics_span: Span,
    pub name: &'a str,
    pub kind: &'static str,
    pub num: usize,
    pub actual_num: usize,
    pub at_least: bool,
}

#[derive(Diagnostic)]
#[diag(resolve_unknown_external_lang_item, code = "E0264")]
pub struct UnknownExternLangItem {
    #[primary_span]
    pub span: Span,
    pub lang_item: Symbol,
}

#[derive(Diagnostic)]
#[diag(resolve_unknown_lang_item, code = "E0522")]
pub struct UnknownLangItem {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}
