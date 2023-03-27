use crate::{errors, structured_errors::StructuredDiagnostic};
use rustc_errors::{DiagnosticBuilder, DiagnosticId, ErrorGuaranteed};
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_session::Session;
use rustc_span::Span;

pub struct MissingCastForVariadicArg<'tcx, 's> {
    pub sess: &'tcx Session,
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub cast_ty: &'s str,
}

impl<'tcx> StructuredDiagnostic<'tcx> for MissingCastForVariadicArg<'tcx, '_> {
    fn session(&self) -> &Session {
        self.sess
    }

    fn code(&self) -> DiagnosticId {
        rustc_errors::error_code!(E0617)
    }

    fn diagnostic_common(&self) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let (sugg_span, replace, help) =
            if let Ok(snippet) = self.sess.source_map().span_to_snippet(self.span) {
                (Some(self.span), format!("{} as {}", snippet, self.cast_ty), None)
            } else {
                (None, "".to_string(), Some(()))
            };

        let mut err = self.sess.create_err(errors::PassToVariadicFunction {
            span: self.span,
            ty: self.ty,
            cast_ty: self.cast_ty,
            help,
            replace,
            sugg_span,
        });

        if self.ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        err
    }

    fn diagnostic_extended(
        &self,
        mut err: DiagnosticBuilder<'tcx, ErrorGuaranteed>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        err.note(&format!(
            "certain types, like `{}`, must be casted before passing them to a \
                variadic function, because of arcane ABI rules dictated by the C \
                standard",
            self.ty
        ));

        err
    }
}
