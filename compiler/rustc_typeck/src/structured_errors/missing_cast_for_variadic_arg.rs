use crate::structured_errors::StructuredDiagnostic;
use rustc_errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_middle::ty::{Ty, TypeFoldable};
use rustc_session::Session;
use rustc_span::Span;

pub struct MissingCastForVariadicArg<'tcx> {
    pub sess: &'tcx Session,
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub cast_ty: &'tcx str,
}

impl<'tcx> StructuredDiagnostic<'tcx> for MissingCastForVariadicArg<'tcx> {
    fn session(&self) -> &Session {
        self.sess
    }

    fn code(&self) -> DiagnosticId {
        rustc_errors::error_code!(E0617)
    }

    fn diagnostic_common(&self) -> DiagnosticBuilder<'tcx> {
        let mut err = if self.ty.references_error() {
            self.sess.diagnostic().struct_dummy()
        } else {
            self.sess.struct_span_fatal_with_code(
                self.span,
                &format!("can't pass `{}` to variadic function", self.ty),
                self.code(),
            )
        };

        if let Ok(snippet) = self.sess.source_map().span_to_snippet(self.span) {
            err.span_suggestion(
                self.span,
                &format!("cast the value to `{}`", self.cast_ty),
                format!("{} as {}", snippet, self.cast_ty),
                Applicability::MachineApplicable,
            );
        } else {
            err.help(&format!("cast the value to `{}`", self.cast_ty));
        }

        err
    }

    fn diagnostic_extended(&self, mut err: DiagnosticBuilder<'tcx>) -> DiagnosticBuilder<'tcx> {
        err.note(&format!(
            "certain types, like `{}`, must be casted before passing them to a \
                variadic function, because of arcane ABI rules dictated by the C \
                standard",
            self.ty
        ));

        err
    }
}
