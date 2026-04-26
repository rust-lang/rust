use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag("unused attribute")]
pub struct UnusedDuplicate {
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    #[warning(
        "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
    )]
    pub warning: bool,
}
