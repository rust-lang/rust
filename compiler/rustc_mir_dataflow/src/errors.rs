use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::path_must_end_in_filename)]
pub(crate) struct PathMustEndInFilename {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::unknown_formatter)]
pub(crate) struct UnknownFormatter {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::duplicate_values_for)]
pub(crate) struct DuplicateValuesFor {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::requires_an_argument)]
pub(crate) struct RequiresAnArgument {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::stop_after_dataflow_ended_compilation)]
pub(crate) struct StopAfterDataFlowEndedCompilation;

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::peek_must_be_place_or_ref_place)]
pub(crate) struct PeekMustBePlaceOrRefPlace {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::peek_must_be_not_temporary)]
pub(crate) struct PeekMustBeNotTemporary {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::peek_bit_not_set)]
pub(crate) struct PeekBitNotSet {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::peek_argument_not_a_local)]
pub(crate) struct PeekArgumentNotALocal {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(mir_dataflow::peek_argument_untracked)]
pub(crate) struct PeekArgumentUntracked {
    #[primary_span]
    pub span: Span,
}
