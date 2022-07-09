use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(restrictions_impl_of_restricted_trait)]
pub struct ImplOfRestrictedTrait {
    #[primary_span]
    pub impl_span: Span,
    #[note]
    pub restriction_span: Span,
}
