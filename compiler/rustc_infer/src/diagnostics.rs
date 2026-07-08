use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag("opaque type's hidden type cannot be another opaque type from the same scope")]
pub(crate) struct OpaqueHiddenTypeDiag {
    #[primary_span]
    #[label("one of the two opaque types used here has to be outside its defining scope")]
    pub span: Span,
    #[note("opaque type whose hidden type is being assigned")]
    pub opaque_type: Span,
    #[note("opaque type being used as hidden type")]
    pub hidden_type: Span,
}
