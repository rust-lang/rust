use crate as rustc_session;
use crate::cgu_reuse_tracker::CguReuse;
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[error(session::incorrect_cgu_reuse_type)]
pub struct IncorrectCguReuseType<'a> {
    #[primary_span]
    pub span: Span,
    pub cgu_user_name: &'a str,
    pub actual_reuse: CguReuse,
    pub expected_reuse: CguReuse,
    pub at_least: &'a str,
}

// #[derive(SessionDiagnostic)]
// #[fatal(session::cgu_not_recorded)]
// pub struct CguNotRecorded<'a> {
//     pub cgu_user_name: &'a str,
//     pub cgu_name: &'a str,
// }
