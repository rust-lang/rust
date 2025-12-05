use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(mir_dataflow_path_must_end_in_filename)]
pub(crate) struct PathMustEndInFilename {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_unknown_formatter)]
pub(crate) struct UnknownFormatter {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_duplicate_values_for)]
pub(crate) struct DuplicateValuesFor {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_requires_an_argument)]
pub(crate) struct RequiresAnArgument {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_stop_after_dataflow_ended_compilation)]
pub(crate) struct StopAfterDataFlowEndedCompilation;

#[derive(Diagnostic)]
#[diag(mir_dataflow_peek_must_be_place_or_ref_place)]
pub(crate) struct PeekMustBePlaceOrRefPlace {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_peek_must_be_not_temporary)]
pub(crate) struct PeekMustBeNotTemporary {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_peek_bit_not_set)]
pub(crate) struct PeekBitNotSet {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_peek_argument_not_a_local)]
pub(crate) struct PeekArgumentNotALocal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_dataflow_peek_argument_untracked)]
pub(crate) struct PeekArgumentUntracked {
    #[primary_span]
    pub span: Span,
}
