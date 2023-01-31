use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(hir_analysis_pattern_type_wild_pat)]
pub struct WildPatTy {
    #[primary_span]
    pub span: Span,
}
