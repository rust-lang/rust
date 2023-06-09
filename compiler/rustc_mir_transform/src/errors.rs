use rustc_errors::{
    DecorateLint, DiagnosticBuilder, DiagnosticMessage, EmissionGuarantee, Handler, IntoDiagnostic,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::{AssertKind, UnsafetyViolationDetails};
use rustc_session::lint::{self, Lint};
use rustc_span::Span;

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
#[diag(mir_transform_unaligned_packed_ref, code = "E0793")]
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
impl<'sess, G: EmissionGuarantee> IntoDiagnostic<'sess, G> for RequiresUnsafe {
    #[track_caller]
    fn into_diagnostic(self, handler: &'sess Handler) -> DiagnosticBuilder<'sess, G> {
        let mut diag =
            handler.struct_diagnostic(crate::fluent_generated::mir_transform_requires_unsafe);
        diag.code(rustc_errors::DiagnosticId::Error("E0133".to_string()));
        diag.set_span(self.span);
        diag.span_label(self.span, self.details.label());
        diag.note(self.details.note());
        let desc = handler.eagerly_translate_to_string(self.details.label(), [].into_iter());
        diag.set_arg("details", desc);
        diag.set_arg("op_in_unsafe_fn_allowed", self.op_in_unsafe_fn_allowed);
        if let Some(sp) = self.enclosing {
            diag.span_label(sp, crate::fluent_generated::mir_transform_not_inherited);
        }
        diag
    }
}

#[derive(Copy, Clone)]
pub(crate) struct RequiresUnsafeDetail {
    pub span: Span,
    pub violation: UnsafetyViolationDetails,
}

impl RequiresUnsafeDetail {
    fn note(self) -> DiagnosticMessage {
        use UnsafetyViolationDetails::*;
        match self.violation {
            CallToUnsafeFunction => crate::fluent_generated::mir_transform_call_to_unsafe_note,
            UseOfInlineAssembly => crate::fluent_generated::mir_transform_use_of_asm_note,
            InitializingTypeWith => {
                crate::fluent_generated::mir_transform_initializing_valid_range_note
            }
            CastOfPointerToInt => crate::fluent_generated::mir_transform_const_ptr2int_note,
            UseOfMutableStatic => crate::fluent_generated::mir_transform_use_of_static_mut_note,
            UseOfExternStatic => crate::fluent_generated::mir_transform_use_of_extern_static_note,
            DerefOfRawPointer => crate::fluent_generated::mir_transform_deref_ptr_note,
            AccessToUnionField => crate::fluent_generated::mir_transform_union_access_note,
            MutationOfLayoutConstrainedField => {
                crate::fluent_generated::mir_transform_mutation_layout_constrained_note
            }
            BorrowOfLayoutConstrainedField => {
                crate::fluent_generated::mir_transform_mutation_layout_constrained_borrow_note
            }
            CallToFunctionWith => crate::fluent_generated::mir_transform_target_feature_call_note,
        }
    }

    fn label(self) -> DiagnosticMessage {
        use UnsafetyViolationDetails::*;
        match self.violation {
            CallToUnsafeFunction => crate::fluent_generated::mir_transform_call_to_unsafe_label,
            UseOfInlineAssembly => crate::fluent_generated::mir_transform_use_of_asm_label,
            InitializingTypeWith => {
                crate::fluent_generated::mir_transform_initializing_valid_range_label
            }
            CastOfPointerToInt => crate::fluent_generated::mir_transform_const_ptr2int_label,
            UseOfMutableStatic => crate::fluent_generated::mir_transform_use_of_static_mut_label,
            UseOfExternStatic => crate::fluent_generated::mir_transform_use_of_extern_static_label,
            DerefOfRawPointer => crate::fluent_generated::mir_transform_deref_ptr_label,
            AccessToUnionField => crate::fluent_generated::mir_transform_union_access_label,
            MutationOfLayoutConstrainedField => {
                crate::fluent_generated::mir_transform_mutation_layout_constrained_label
            }
            BorrowOfLayoutConstrainedField => {
                crate::fluent_generated::mir_transform_mutation_layout_constrained_borrow_label
            }
            CallToFunctionWith => crate::fluent_generated::mir_transform_target_feature_call_label,
        }
    }
}

pub(crate) struct UnsafeOpInUnsafeFn {
    pub details: RequiresUnsafeDetail,
}

impl<'a> DecorateLint<'a, ()> for UnsafeOpInUnsafeFn {
    #[track_caller]
    fn decorate_lint<'b>(
        self,
        diag: &'b mut DiagnosticBuilder<'a, ()>,
    ) -> &'b mut DiagnosticBuilder<'a, ()> {
        let desc = diag
            .handler()
            .expect("lint should not yet be emitted")
            .eagerly_translate_to_string(self.details.label(), [].into_iter());
        diag.set_arg("details", desc);
        diag.span_label(self.details.span, self.details.label());
        diag.note(self.details.note());
        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        crate::fluent_generated::mir_transform_unsafe_op_in_unsafe_fn
    }
}

pub(crate) enum AssertLint<P> {
    ArithmeticOverflow(Span, AssertKind<P>),
    UnconditionalPanic(Span, AssertKind<P>),
}

impl<'a, P: std::fmt::Debug> DecorateLint<'a, ()> for AssertLint<P> {
    fn decorate_lint<'b>(
        self,
        diag: &'b mut DiagnosticBuilder<'a, ()>,
    ) -> &'b mut DiagnosticBuilder<'a, ()> {
        let span = self.span();
        let assert_kind = self.panic();
        let message = assert_kind.diagnostic_message();
        assert_kind.add_args(&mut |name, value| {
            diag.set_arg(name, value);
        });
        diag.span_label(span, message);

        diag
    }

    fn msg(&self) -> DiagnosticMessage {
        match self {
            AssertLint::ArithmeticOverflow(..) => {
                crate::fluent_generated::mir_transform_arithmetic_overflow
            }
            AssertLint::UnconditionalPanic(..) => {
                crate::fluent_generated::mir_transform_operation_will_panic
            }
        }
    }
}

impl<P> AssertLint<P> {
    pub fn lint(&self) -> &'static Lint {
        match self {
            AssertLint::ArithmeticOverflow(..) => lint::builtin::ARITHMETIC_OVERFLOW,
            AssertLint::UnconditionalPanic(..) => lint::builtin::UNCONDITIONAL_PANIC,
        }
    }
    pub fn span(&self) -> Span {
        match self {
            AssertLint::ArithmeticOverflow(sp, _) | AssertLint::UnconditionalPanic(sp, _) => *sp,
        }
    }
    pub fn panic(self) -> AssertKind<P> {
        match self {
            AssertLint::ArithmeticOverflow(_, p) | AssertLint::UnconditionalPanic(_, p) => p,
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
#[diag(mir_transform_must_not_suspend)]
pub(crate) struct MustNotSupend<'a> {
    #[label]
    pub yield_sp: Span,
    #[subdiagnostic]
    pub reason: Option<MustNotSuspendReason>,
    #[help]
    pub src_sp: Span,
    pub pre: &'a str,
    pub def_path: String,
    pub post: &'a str,
}

#[derive(Subdiagnostic)]
#[note(mir_transform_note)]
pub(crate) struct MustNotSuspendReason {
    #[primary_span]
    pub span: Span,
    pub reason: String,
}

#[derive(Diagnostic)]
#[diag(mir_transform_simd_shuffle_last_const)]
pub(crate) struct SimdShuffleLastConst {
    #[primary_span]
    pub span: Span,
}
