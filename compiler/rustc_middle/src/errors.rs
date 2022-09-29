use rustc_errors::{fluent, AddSubdiagnostic, ErrorGuaranteed};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_session::{lint::Level, parse::ParseSess, SessionDiagnostic};
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[error(code = "E0320", slug = "overflow-while-adding-drop-check-rules")]
pub struct DropckOutlivesErr {
    span: Span,
    overflow_ty: Ty<'tcx>,
}
