use rustc_errors::{
    codes::*, Applicability, Diag, DiagMessage, EmissionGuarantee, LintDiagnostic,
    SubdiagMessageOp, Subdiagnostic,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::AssertKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{self, Lint};
use rustc_span::def_id::DefId;
use rustc_span::Span;

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
        let message = self.assert_kind.diagnostic_message();
        self.assert_kind.add_args(&mut |name, value| {
            diag.arg(name, value);
        });
        diag.span_label(self.span, message);
    }

    fn msg(&self) -> DiagMessage {
        match self.lint_kind {
            AssertLintKind::ArithmeticOverflow => fluent::mir_transform_arithmetic_overflow,
            AssertLintKind::UnconditionalPanic => fluent::mir_transform_operation_will_panic,
        }
    }
}

impl AssertLintKind {
    pub fn lint(&self) -> &'static Lint {
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

#[derive(LintDiagnostic)]
#[diag(mir_transform_unused_capture_maybe_capture_ref)]
#[help]
pub(crate) struct UnusedCaptureMaybeCaptureRef {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_unused_var_assigned_only)]
#[note]
pub(crate) struct UnusedVarAssignedOnly {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_unused_assign)]
#[help]
pub(crate) struct UnusedAssign {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_unused_assign_passed)]
#[help]
pub(crate) struct UnusedAssignPassed {
    pub name: String,
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_unused_variable)]
pub(crate) struct UnusedVariable {
    pub name: String,
    #[subdiagnostic]
    pub string_interp: Vec<UnusedVariableStringInterp>,
    #[subdiagnostic]
    pub sugg: UnusedVariableSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedVariableSugg {
    #[multipart_suggestion(
        mir_transform_unused_variable_try_ignore,
        applicability = "machine-applicable"
    )]
    TryIgnore {
        #[suggestion_part(code = "{name}: _")]
        shorthands: Vec<Span>,
        #[suggestion_part(code = "_")]
        non_shorthands: Vec<Span>,
        name: String,
    },

    #[multipart_suggestion(
        mir_transform_unused_var_underscore,
        applicability = "machine-applicable"
    )]
    TryPrefix {
        #[suggestion_part(code = "_{name}")]
        spans: Vec<Span>,
        name: String,
    },

    #[help(mir_transform_unused_variable_args_in_macro)]
    NoSugg {
        #[primary_span]
        span: Span,
        name: String,
    },
}

pub(crate) struct UnusedVariableStringInterp {
    pub lit: Span,
}

impl Subdiagnostic for UnusedVariableStringInterp {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: F,
    ) {
        diag.span_label(
            self.lit,
            crate::fluent_generated::mir_transform_maybe_string_interpolation,
        );
        diag.multipart_suggestion(
            crate::fluent_generated::mir_transform_string_interpolation_only_works,
            vec![
                (self.lit.shrink_to_lo(), String::from("format!(")),
                (self.lit.shrink_to_hi(), String::from(")")),
            ],
            Applicability::MachineApplicable,
        );
    }
}

pub(crate) struct MustNotSupend<'tcx, 'a> {
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
        diag.span_label(self.yield_sp, fluent::_subdiag::label);
        if let Some(reason) = self.reason {
            diag.subdiagnostic(diag.dcx, reason);
        }
        diag.span_help(self.src_sp, fluent::_subdiag::help);
        diag.arg("pre", self.pre);
        diag.arg("def_path", self.tcx.def_path_str(self.def_id));
        diag.arg("post", self.post);
    }

    fn msg(&self) -> rustc_errors::DiagMessage {
        fluent::mir_transform_must_not_suspend
    }
}

#[derive(Subdiagnostic)]
#[note(mir_transform_note)]
pub(crate) struct MustNotSuspendReason {
    #[primary_span]
    pub span: Span,
    pub reason: String,
}
