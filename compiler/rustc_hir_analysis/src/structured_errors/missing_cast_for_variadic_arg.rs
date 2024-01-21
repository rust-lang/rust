use crate::{errors, structured_errors::StructuredDiagnostic};
use rustc_errors::{codes::*, Applicability, DiagnosticBuilder, ErrCode};
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

    fn code(&self) -> ErrCode {
        E0617
    }

    fn diagnostic_common(&self) -> DiagnosticBuilder<'tcx> {
        let (sugg_span, replace, help) =
            if let Ok(snippet) = self.sess.source_map().span_to_snippet(self.span) {
                (Some(self.span), format!("{} as {}", snippet, self.cast_ty), None)
            } else {
                (None, "".to_string(), Some(()))
            };

        let mut err = self.sess.dcx().create_err(errors::PassToVariadicFunction {
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

        let msg = if self.ty.is_fn() {
            err.help("a function item is zero-sized and needs to be casted into a function pointer to be used in FFI")
                .note("for more information on function items, visit https://doc.rust-lang.org/reference/types/function-item.html");
            "cast the value into a function pointer".to_string()
        } else {
            format!("cast the value to `{}`", self.cast_ty)
        };
        err.span_suggestion_verbose(
            self.span.shrink_to_hi(),
            msg,
            format!(" as {}", self.cast_ty),
            Applicability::MachineApplicable,
        );

        err
    }

    fn diagnostic_extended(&self, mut err: DiagnosticBuilder<'tcx>) -> DiagnosticBuilder<'tcx> {
        err.note(format!(
            "certain types, like `{}`, must be casted before passing them to a \
                variadic function, because of arcane ABI rules dictated by the C \
                standard",
            self.ty
        ));

        err
    }
}
