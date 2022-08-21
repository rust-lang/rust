use rustc_hir::ConstContext;
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(const_eval::unstable_in_stable)]
pub(crate) struct UnstableInStable {
    pub gate: String,
    #[primary_span]
    pub span: Span,
    #[suggestion(
        const_eval::unstable_sugg,
        code = "#[rustc_const_unstable(feature = \"...\", issue = \"...\")]\n",
        applicability = "has-placeholders"
    )]
    #[suggestion(
        const_eval::bypass_sugg,
        code = "#[rustc_allow_const_fn_unstable({gate})]\n",
        applicability = "has-placeholders"
    )]
    pub attr_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::thread_local_access, code = "E0625")]
pub(crate) struct NonConstOpErr {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::static_access, code = "E0013")]
#[help]
pub(crate) struct StaticAccessErr {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[note(const_eval::teach_note)]
    #[help(const_eval::teach_help)]
    pub teach: Option<()>,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::raw_ptr_to_int)]
#[note]
#[note(const_eval::note2)]
pub(crate) struct RawPtrToIntErr {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::raw_ptr_comparison)]
#[note]
pub(crate) struct RawPtrComparisonErr {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::panic_non_str)]
pub(crate) struct PanicNonStrErr {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::mut_deref, code = "E0658")]
pub(crate) struct MutDerefErr {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::transient_mut_borrow, code = "E0658")]
pub(crate) struct TransientMutBorrowErr {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(SessionDiagnostic)]
#[diag(const_eval::transient_mut_borrow_raw, code = "E0658")]
pub(crate) struct TransientMutBorrowErrRaw {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}
