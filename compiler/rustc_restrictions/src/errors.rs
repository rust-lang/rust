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

#[derive(Diagnostic)]
#[diag(restrictions_mut_of_restricted_field)]
pub struct MutOfRestrictedField {
    #[primary_span]
    pub mut_span: Span,
    #[note]
    pub restriction_span: Span,
}

#[derive(Diagnostic)]
#[diag(restrictions_construction_of_ty_with_mut_restricted_field)]
pub struct ConstructionOfTyWithMutRestrictedField {
    #[primary_span]
    pub construction_span: Span,
    #[label]
    pub restriction_span: Span,
    #[note]
    pub note: (),
    pub ty: &'static str,
}
