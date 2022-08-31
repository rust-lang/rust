use std::num::NonZeroU32;

use crate as rustc_session;
use crate::cgu_reuse_tracker::CguReuse;
use rustc_errors::MultiSpan;
use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[diag(session::incorrect_cgu_reuse_type)]
pub struct IncorrectCguReuseType<'a> {
    #[primary_span]
    pub span: Span,
    pub cgu_user_name: &'a str,
    pub actual_reuse: CguReuse,
    pub expected_reuse: CguReuse,
    pub at_least: u8,
}

#[derive(SessionDiagnostic)]
#[diag(session::cgu_not_recorded)]
pub struct CguNotRecorded<'a> {
    pub cgu_user_name: &'a str,
    pub cgu_name: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(session::feature_gate_error, code = "E0658")]
pub struct FeatureGateError<'a> {
    #[primary_span]
    pub span: MultiSpan,
    pub explain: &'a str,
}

#[derive(SessionSubdiagnostic)]
#[note(session::feature_diagnostic_for_issue)]
pub struct FeatureDiagnosticForIssue {
    pub n: NonZeroU32,
}

#[derive(SessionSubdiagnostic)]
#[help(session::feature_diagnostic_help)]
pub struct FeatureDiagnosticHelp {
    pub feature: Symbol,
}
