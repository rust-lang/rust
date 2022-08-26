use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

use crate::ty::Ty;

#[derive(SessionDiagnostic)]
#[diag(middle::drop_check_overflow, code = "E0320")]
#[note]
pub struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub note: String,
}
