use rustc::session::Session;
use syntax_pos::Span;
use errors::{Applicability, DiagnosticId, DiagnosticBuilder};
use rustc::ty::{Ty, TypeFoldable};

pub trait StructuredDiagnostic<'tcx> {
    fn session(&self) -> &Session;

    fn code(&self) -> DiagnosticId;

    fn common(&self) -> DiagnosticBuilder<'tcx>;

    fn diagnostic(&self) -> DiagnosticBuilder<'tcx> {
        let err = self.common();
        if self.session().teach(&self.code()) {
            self.extended(err)
        } else {
            self.regular(err)
        }
    }

    fn regular(&self, err: DiagnosticBuilder<'tcx>) -> DiagnosticBuilder<'tcx> {
        err
    }

    fn extended(&self, err: DiagnosticBuilder<'tcx>) -> DiagnosticBuilder<'tcx> {
        err
    }
}

pub struct VariadicError<'tcx> {
    sess: &'tcx Session,
    span: Span,
    t: Ty<'tcx>,
    cast_ty: &'tcx str,
}

impl<'tcx> VariadicError<'tcx> {
    pub fn new(sess: &'tcx Session,
               span: Span,
               t: Ty<'tcx>,
               cast_ty: &'tcx str) -> VariadicError<'tcx> {
        VariadicError { sess, span, t, cast_ty }
    }
}

impl<'tcx> StructuredDiagnostic<'tcx> for VariadicError<'tcx> {
    fn session(&self) -> &Session { self.sess }

    fn code(&self) -> DiagnosticId {
        __diagnostic_used!(E0617);
        DiagnosticId::Error("E0617".to_owned())
    }

    fn common(&self) -> DiagnosticBuilder<'tcx> {
        let mut err = if self.t.references_error() {
            self.sess.diagnostic().struct_dummy()
        } else {
            self.sess.struct_span_fatal_with_code(
                self.span,
                &format!("can't pass `{}` to variadic function", self.t),
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

    fn extended(&self, mut err: DiagnosticBuilder<'tcx>) -> DiagnosticBuilder<'tcx> {
        err.note(&format!("certain types, like `{}`, must be cast before passing them to a \
                           variadic function, because of arcane ABI rules dictated by the C \
                           standard",
                          self.t));
        err
    }
}

pub struct SizedUnsizedCastError<'tcx> {
    sess: &'tcx Session,
    span: Span,
    expr_ty: Ty<'tcx>,
    cast_ty: String,
}

impl<'tcx> SizedUnsizedCastError<'tcx> {
    pub fn new(sess: &'tcx Session,
               span: Span,
               expr_ty: Ty<'tcx>,
               cast_ty: String) -> SizedUnsizedCastError<'tcx> {
        SizedUnsizedCastError { sess, span, expr_ty, cast_ty }
    }
}

impl<'tcx> StructuredDiagnostic<'tcx> for SizedUnsizedCastError<'tcx> {
    fn session(&self) -> &Session { self.sess }

    fn code(&self) -> DiagnosticId {
        __diagnostic_used!(E0607);
        DiagnosticId::Error("E0607".to_owned())
    }

    fn common(&self) -> DiagnosticBuilder<'tcx> {
        if self.expr_ty.references_error() {
            self.sess.diagnostic().struct_dummy()
        } else {
            self.sess.struct_span_fatal_with_code(
                self.span,
                &format!("cannot cast thin pointer `{}` to fat pointer `{}`",
                         self.expr_ty,
                         self.cast_ty),
                self.code(),
            )
        }
    }

    fn extended(&self, mut err: DiagnosticBuilder<'tcx>) -> DiagnosticBuilder<'tcx> {
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
https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions");
        err
    }
}
