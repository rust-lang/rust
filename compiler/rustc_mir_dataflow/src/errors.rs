use rustc_macros::Diagnostic;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag("stop_after_dataflow ended compilation")]
pub(crate) struct StopAfterDataFlowEndedCompilation;

#[derive(Diagnostic)]
#[diag("rustc_peek: argument expression must be either `place` or `&place`")]
pub(crate) struct PeekMustBePlaceOrRefPlace {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("dataflow::sanity_check cannot feed a non-temp to rustc_peek")]
pub(crate) struct PeekMustBeNotTemporary {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("rustc_peek: bit not set")]
pub(crate) struct PeekBitNotSet {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("rustc_peek: argument was not a local")]
pub(crate) struct PeekArgumentNotALocal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("rustc_peek: argument untracked")]
pub(crate) struct PeekArgumentUntracked {
    #[primary_span]
    pub span: Span,
}
