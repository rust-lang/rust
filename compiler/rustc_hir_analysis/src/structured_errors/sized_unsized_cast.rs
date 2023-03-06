use crate::structured_errors::StructuredDiagnostic;
use rustc_errors::{DiagnosticBuilder, DiagnosticId, ErrorGuaranteed};
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_session::Session;
use rustc_span::Span;

pub struct SizedUnsizedCast<'tcx> {
    pub sess: &'tcx Session,
    pub span: Span,
    pub expr_ty: Ty<'tcx>,
    pub cast_ty: String,
}

impl<'tcx> StructuredDiagnostic<'tcx> for SizedUnsizedCast<'tcx> {
    fn session(&self) -> &Session {
        self.sess
    }

    fn code(&self) -> DiagnosticId {
        rustc_errors::error_code!(E0607)
    }

    fn diagnostic_common(&self) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = self.sess.struct_span_err_with_code(
            self.span,
            &format!(
                "cannot cast thin pointer `{}` to fat pointer `{}`",
                self.expr_ty, self.cast_ty
            ),
            self.code(),
        );

        if self.expr_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        err
    }

    fn diagnostic_extended(
        &self,
        mut err: DiagnosticBuilder<'tcx, ErrorGuaranteed>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        err.help(
            "Thin pointers are \"simple\" pointers: they are purely a reference to a
memory address.

Fat pointers are pointers referencing \"Dynamically Sized Types\" (also
called DST). DST don't have a statically known size, therefore they can
only exist behind some kind of pointers that contain additional
information. Slices and trait objects are DSTs. In the case of slices,
the additional information the fat pointer holds is their size.

To fix this error, don't try to cast directly between thin and fat
pointers.

For more information about casts, take a look at The Book:
https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions",
        );
        err
    }
}
