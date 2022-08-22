use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[error(driver::rlink_unable_to_read)]
pub(crate) struct RlinkUnableToRead {
    #[primary_span]
    pub span: Span,
    pub error_message: String,
}

#[derive(SessionDiagnostic)]
#[error(driver::rlink_unable_to_deserialize)]
pub(crate) struct RlinkUnableToDeserialize {
    #[primary_span]
    pub span: Span,
    pub error_message: String,
}

#[derive(SessionDiagnostic)]
#[error(driver::rlink_no_a_file)]
pub(crate) struct RlinkNotAFile {
    #[primary_span]
    pub span: Span,
}
