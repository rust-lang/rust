use rustc_errors::codes::*;
use rustc_errors::{Diag, LintDiagnostic};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::AssertKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{self, Lint};
use rustc_span::Span;
use rustc_span::def_id::DefId;

use crate::fluent_generated as fluent;

#[derive(LintDiagnostic)]
pub(crate) enum ConstMutate {
    #[diag(mir_transform_const_modify)]
    #[note]
    Modify {
        #[note(mir_transform_const_defined_here)]
        konst: Span,
    },
    #[diag(mir_transform_const_mut_borrow)]
    #[note]
    #[note(mir_transform_note2)]
    MutBorrow {
        #[note(mir_transform_note3)]
        method_call: Option<Span>,
        #[note(mir_transform_const_defined_here)]
        konst: Span,
    },
}

#[derive(Diagnostic)]
#[diag(mir_transform_unaligned_packed_ref, code = E0793)]
#[note]
#[note(mir_transform_note_ub)]
#[help]
pub(crate) struct UnalignedPackedRef {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_transform_unknown_pass_name)]
pub(crate) struct UnknownPassName<'a> {
    pub(crate) name: &'a str,
}

pub(crate) struct AssertLint<P> {
    pub span: Span,
    pub assert_kind: AssertKind<P>,
    pub lint_kind: AssertLintKind,
}

pub(crate) enum AssertLintKind {
    ArithmeticOverflow,
    UnconditionalPanic,
}

impl<'a, P: std::fmt::Debug> LintDiagnostic<'a, ()> for AssertLint<P> {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(match self.lint_kind {
            AssertLintKind::ArithmeticOverflow => fluent::mir_transform_arithmetic_overflow,
            AssertLintKind::UnconditionalPanic => fluent::mir_transform_operation_will_panic,
        });
        let label = self.assert_kind.diagnostic_message();
        self.assert_kind.add_args(&mut |name, value| {
            diag.arg(name, value);
        });
        diag.span_label(self.span, label);
    }
}

impl AssertLintKind {
    pub(crate) fn lint(&self) -> &'static Lint {
        match self {
            AssertLintKind::ArithmeticOverflow => lint::builtin::ARITHMETIC_OVERFLOW,
            AssertLintKind::UnconditionalPanic => lint::builtin::UNCONDITIONAL_PANIC,
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_ffi_unwind_call)]
pub(crate) struct FfiUnwindCall {
    #[label(mir_transform_ffi_unwind_call)]
    pub span: Span,
    pub foreign: bool,
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_fn_item_ref)]
pub(crate) struct FnItemRef {
    #[suggestion(code = "{sugg}", applicability = "unspecified")]
    pub span: Span,
    pub sugg: String,
    pub ident: String,
}

#[derive(Diagnostic)]
#[diag(mir_transform_exceeds_mcdc_test_vector_limit)]
pub(crate) struct MCDCExceedsTestVectorLimit {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) max_num_test_vectors: usize,
}

pub(crate) struct MustNotSupend<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub yield_sp: Span,
    pub reason: Option<MustNotSuspendReason>,
    pub src_sp: Span,
    pub pre: &'a str,
    pub def_id: DefId,
    pub post: &'a str,
}

// Needed for def_path_str
impl<'a> LintDiagnostic<'a, ()> for MustNotSupend<'_, '_> {
    fn decorate_lint<'b>(self, diag: &'b mut rustc_errors::Diag<'a, ()>) {
        diag.primary_message(fluent::mir_transform_must_not_suspend);
        diag.span_label(self.yield_sp, fluent::_subdiag::label);
        if let Some(reason) = self.reason {
            diag.subdiagnostic(reason);
        }
        diag.span_help(self.src_sp, fluent::_subdiag::help);
        diag.arg("pre", self.pre);
        diag.arg("def_path", self.tcx.def_path_str(self.def_id));
        diag.arg("post", self.post);
    }
}

#[derive(Subdiagnostic)]
#[note(mir_transform_note)]
pub(crate) struct MustNotSuspendReason {
    #[primary_span]
    pub span: Span,
    pub reason: String,
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_undefined_transmute)]
#[note]
#[note(mir_transform_note2)]
#[help]
pub(crate) struct UndefinedTransmute;
