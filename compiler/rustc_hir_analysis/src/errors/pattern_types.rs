use rustc_macros::Diagnostic;
use rustc_middle::ty::Ty;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(hir_analysis_invalid_base_type)]
pub(crate) struct InvalidBaseType<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    pub ty_span: Span,
    pub pat: &'static str,
    #[note]
    pub pat_span: Span,
}
