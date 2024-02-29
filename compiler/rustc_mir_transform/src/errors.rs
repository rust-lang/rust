use std::borrow::Cow;

use rustc_errors::{
    codes::*, Applicability, DecorateLint, Diag, DiagArgValue, DiagCtxt, DiagMessage,
    EmissionGuarantee, IntoDiagnostic, Level,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::{AssertKind, UnsafetyViolationDetails};
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

#[derive(LintDiagnostic)]
#[diag(mir_transform_unused_unsafe)]
pub(crate) struct UnusedUnsafe {
    #[label(mir_transform_unused_unsafe)]
    pub span: Span,
    #[label]
    pub nested_parent: Option<Span>,
}

pub(crate) struct RequiresUnsafe {
    pub span: Span,
    pub details: RequiresUnsafeDetail,
    pub enclosing: Option<Span>,
    pub op_in_unsafe_fn_allowed: bool,
}

// The primary message for this diagnostic should be '{$label} is unsafe and...',
// so we need to eagerly translate the label here, which isn't supported by the derive API
// We could also exhaustively list out the primary messages for all unsafe violations,
// but this would result in a lot of duplication.
impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for RequiresUnsafe {
    #[track_caller]
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(dcx, level, fluent::mir_transform_requires_unsafe);
        diag.code(E0133);
        diag.span(self.span);
        diag.span_label(self.span, self.details.label());
        let desc = dcx.eagerly_translate_to_string(self.details.label(), [].into_iter());
        diag.arg("details", desc);
        diag.arg("op_in_unsafe_fn_allowed", self.op_in_unsafe_fn_allowed);
        self.details.add_subdiagnostics(&mut diag);
        if let Some(sp) = self.enclosing {
            diag.span_label(sp, fluent::mir_transform_not_inherited);
        }
        diag
    }
}

#[derive(Clone)]
pub(crate) struct RequiresUnsafeDetail {
    pub span: Span,
    pub violation: UnsafetyViolationDetails,
}

impl RequiresUnsafeDetail {
    // FIXME: make this translatable
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)]
    fn add_subdiagnostics<G: EmissionGuarantee>(&self, diag: &mut Diag<'_, G>) {
        use UnsafetyViolationDetails::*;
        match self.violation {
            CallToUnsafeFunction => {
                diag.note(fluent::mir_transform_call_to_unsafe_note);
            }
            UseOfInlineAssembly => {
                diag.note(fluent::mir_transform_use_of_asm_note);
            }
            InitializingTypeWith => {
                diag.note(fluent::mir_transform_initializing_valid_range_note);
            }
            CastOfPointerToInt => {
                diag.note(fluent::mir_transform_const_ptr2int_note);
            }
            UseOfMutableStatic => {
                diag.note(fluent::mir_transform_use_of_static_mut_note);
            }
            UseOfExternStatic => {
                diag.note(fluent::mir_transform_use_of_extern_static_note);
            }
            DerefOfRawPointer => {
                diag.note(fluent::mir_transform_deref_ptr_note);
            }
            AccessToUnionField => {
                diag.note(fluent::mir_transform_union_access_note);
            }
            MutationOfLayoutConstrainedField => {
                diag.note(fluent::mir_transform_mutation_layout_constrained_note);
            }
            BorrowOfLayoutConstrainedField => {
                diag.note(fluent::mir_transform_mutation_layout_constrained_borrow_note);
            }
            CallToFunctionWith { ref missing, ref build_enabled } => {
                diag.help(fluent::mir_transform_target_feature_call_help);
                diag.arg(
                    "missing_target_features",
                    DiagArgValue::StrListSepByAnd(
                        missing.iter().map(|feature| Cow::from(feature.to_string())).collect(),
                    ),
                );
                diag.arg("missing_target_features_count", missing.len());
                if !build_enabled.is_empty() {
                    diag.note(fluent::mir_transform_target_feature_call_note);
                    diag.arg(
                        "build_target_features",
                        DiagArgValue::StrListSepByAnd(
                            build_enabled
                                .iter()
                                .map(|feature| Cow::from(feature.to_string()))
                                .collect(),
                        ),
                    );
                    diag.arg("build_target_features_count", build_enabled.len());
                }
            }
        }
    }

    fn label(&self) -> DiagMessage {
        use UnsafetyViolationDetails::*;
        match self.violation {
            CallToUnsafeFunction => fluent::mir_transform_call_to_unsafe_label,
            UseOfInlineAssembly => fluent::mir_transform_use_of_asm_label,
            InitializingTypeWith => fluent::mir_transform_initializing_valid_range_label,
            CastOfPointerToInt => fluent::mir_transform_const_ptr2int_label,
            UseOfMutableStatic => fluent::mir_transform_use_of_static_mut_label,
            UseOfExternStatic => fluent::mir_transform_use_of_extern_static_label,
            DerefOfRawPointer => fluent::mir_transform_deref_ptr_label,
            AccessToUnionField => fluent::mir_transform_union_access_label,
            MutationOfLayoutConstrainedField => {
                fluent::mir_transform_mutation_layout_constrained_label
            }
            BorrowOfLayoutConstrainedField => {
                fluent::mir_transform_mutation_layout_constrained_borrow_label
            }
            CallToFunctionWith { .. } => fluent::mir_transform_target_feature_call_label,
        }
    }
}

pub(crate) struct UnsafeOpInUnsafeFn {
    pub details: RequiresUnsafeDetail,

    /// These spans point to:
    ///  1. the start of the function body
    ///  2. the end of the function body
    ///  3. the function signature
    pub suggest_unsafe_block: Option<(Span, Span, Span)>,
}

impl<'a> DecorateLint<'a, ()> for UnsafeOpInUnsafeFn {
    #[track_caller]
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        let desc = diag.dcx.eagerly_translate_to_string(self.details.label(), [].into_iter());
        diag.arg("details", desc);
        diag.span_label(self.details.span, self.details.label());
        self.details.add_subdiagnostics(diag);

        if let Some((start, end, fn_sig)) = self.suggest_unsafe_block {
            diag.span_note(fn_sig, fluent::mir_transform_note);
            diag.tool_only_multipart_suggestion(
                fluent::mir_transform_suggestion,
                vec![(start, " unsafe {".into()), (end, "}".into())],
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn msg(&self) -> DiagMessage {
        fluent::mir_transform_unsafe_op_in_unsafe_fn
    }
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

impl<'a, P: std::fmt::Debug> DecorateLint<'a, ()> for AssertLint<P> {
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
impl<'a> DecorateLint<'a, ()> for MustNotSupend<'_, '_> {
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
