use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level,
    MultiSpan, Subdiagnostic, pluralize,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_pattern_analysis::errors::Uncovered;
use rustc_pattern_analysis::rustc::RustcPatCtxt;
use rustc_span::{Ident, Span, Symbol};

use crate::fluent_generated as fluent;

#[derive(LintDiagnostic)]
#[diag(mir_build_call_to_deprecated_safe_fn_requires_unsafe)]
pub(crate) struct CallToDeprecatedSafeFnRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) guarantee: String,
    #[subdiagnostic]
    pub(crate) sub: CallToDeprecatedSafeFnRequiresUnsafeSub,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(mir_build_suggestion, applicability = "machine-applicable")]
pub(crate) struct CallToDeprecatedSafeFnRequiresUnsafeSub {
    pub(crate) start_of_line_suggestion: String,
    #[suggestion_part(code = "{start_of_line_suggestion}")]
    pub(crate) start_of_line: Span,
    #[suggestion_part(code = "unsafe {{ ")]
    pub(crate) left: Span,
    #[suggestion_part(code = " }}")]
    pub(crate) right: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_unsafe_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnInitializingTypeWithUnsafeFieldRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_extern_static_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_unsafe_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnUseOfUnsafeFieldRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_union_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(
    mir_build_unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe,
    code = E0133
)]
#[note]
pub(crate) struct UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(
    mir_build_unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe,
    code = E0133,
)]
pub(crate) struct UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(
    mir_build_unsafe_binder_cast_requires_unsafe,
    code = E0133,
)]
pub(crate) struct UnsafeOpInUnsafeFnUnsafeBinderCastRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe, code = E0133)]
#[help]
pub(crate) struct UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe {
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) missing_target_features: DiagArgValue,
    pub(crate) missing_target_features_count: usize,
    #[note]
    pub(crate) note: bool,
    pub(crate) build_target_features: DiagArgValue,
    pub(crate) build_target_features_count: usize,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedLintNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_nameless, code = E0133)]
#[note]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafeNameless {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed,
    code = E0133
)]
#[note]
pub(crate) struct CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UseOfInlineAssemblyRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_initializing_type_with_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct InitializingTypeWithRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_initializing_type_with_unsafe_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct InitializingTypeWithUnsafeFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = E0133
)]
#[note]
pub(crate) struct InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_initializing_type_with_unsafe_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = E0133
)]
#[note]
pub(crate) struct InitializingTypeWithUnsafeFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutable_static_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UseOfMutableStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_extern_static_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UseOfExternStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_unsafe_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct UseOfUnsafeFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_unsafe_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct UseOfUnsafeFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_deref_raw_pointer_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct DerefOfRawPointerRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_union_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct AccessToUnionFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[note]
pub(crate) struct AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutation_of_layout_constrained_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct MutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = E0133
)]
#[note]
pub(crate) struct MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_borrow_of_layout_constrained_field_requires_unsafe, code = E0133)]
#[note]
pub(crate) struct BorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = E0133
)]
#[note]
pub(crate) struct BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe, code = E0133)]
#[help]
pub(crate) struct CallToFunctionWithRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) missing_target_features: DiagArgValue,
    pub(crate) missing_target_features_count: usize,
    #[note]
    pub(crate) note: bool,
    pub(crate) build_target_features: DiagArgValue,
    pub(crate) build_target_features_count: usize,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = E0133)]
#[help]
pub(crate) struct CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) function: String,
    pub(crate) missing_target_features: DiagArgValue,
    pub(crate) missing_target_features_count: usize,
    #[note]
    pub(crate) note: bool,
    pub(crate) build_target_features: DiagArgValue,
    pub(crate) build_target_features_count: usize,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_unsafe_binder_cast_requires_unsafe,
    code = E0133,
)]
pub(crate) struct UnsafeBinderCastRequiresUnsafe {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_unsafe_binder_cast_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = E0133,
)]
pub(crate) struct UnsafeBinderCastRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Subdiagnostic)]
#[label(mir_build_unsafe_not_inherited)]
pub(crate) struct UnsafeNotInheritedNote {
    #[primary_span]
    pub(crate) span: Span,
}

pub(crate) struct UnsafeNotInheritedLintNote {
    pub(crate) signature_span: Span,
    pub(crate) body_span: Span,
}

impl Subdiagnostic for UnsafeNotInheritedLintNote {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_note(self.signature_span, fluent::mir_build_unsafe_fn_safe_body);
        let body_start = self.body_span.shrink_to_lo();
        let body_end = self.body_span.shrink_to_hi();
        diag.tool_only_multipart_suggestion(
            fluent::mir_build_wrap_suggestion,
            vec![(body_start, "{ unsafe ".into()), (body_end, "}".into())],
            Applicability::MachineApplicable,
        );
    }
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unused_unsafe)]
pub(crate) struct UnusedUnsafe {
    #[label]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) enclosing: Option<UnusedUnsafeEnclosing>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedUnsafeEnclosing {
    #[label(mir_build_unused_unsafe_enclosing_block_label)]
    Block {
        #[primary_span]
        span: Span,
    },
}

pub(crate) struct NonExhaustivePatternsTypeNotEmpty<'p, 'tcx, 'm> {
    pub(crate) cx: &'m RustcPatCtxt<'p, 'tcx>,
    pub(crate) scrut_span: Span,
    pub(crate) braces_span: Option<Span>,
    pub(crate) ty: Ty<'tcx>,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for NonExhaustivePatternsTypeNotEmpty<'_, '_, '_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag =
            Diag::new(dcx, level, fluent::mir_build_non_exhaustive_patterns_type_not_empty);
        diag.span(self.scrut_span);
        diag.code(E0004);
        let peeled_ty = self.ty.peel_refs();
        diag.arg("ty", self.ty);
        diag.arg("peeled_ty", peeled_ty);

        if let ty::Adt(def, _) = peeled_ty.kind() {
            let def_span = self
                .cx
                .tcx
                .hir_get_if_local(def.did())
                .and_then(|node| node.ident())
                .map(|ident| ident.span)
                .unwrap_or_else(|| self.cx.tcx.def_span(def.did()));

            // workaround to make test pass
            let mut span: MultiSpan = def_span.into();
            span.push_span_label(def_span, "");

            diag.span_note(span, fluent::mir_build_def_note);
        }

        let is_non_exhaustive = matches!(self.ty.kind(),
            ty::Adt(def, _) if def.variant_list_has_applicable_non_exhaustive());
        if is_non_exhaustive {
            diag.note(fluent::mir_build_non_exhaustive_type_note);
        } else {
            diag.note(fluent::mir_build_type_note);
        }

        if let ty::Ref(_, sub_ty, _) = self.ty.kind() {
            if !sub_ty.is_inhabited_from(self.cx.tcx, self.cx.module, self.cx.typing_env) {
                diag.note(fluent::mir_build_reference_note);
            }
        }

        let sm = self.cx.tcx.sess.source_map();
        if let Some(braces_span) = self.braces_span {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(self.scrut_span)
            {
                (format!("\n{snippet}"), "    ")
            } else {
                (" ".to_string(), "")
            };
            diag.span_suggestion_verbose(
                braces_span,
                fluent::mir_build_suggestion,
                format!(" {{{indentation}{more}_ => todo!(),{indentation}}}"),
                Applicability::HasPlaceholders,
            );
        } else {
            diag.help(fluent::mir_build_help);
        }

        diag
    }
}

#[derive(Subdiagnostic)]
#[note(mir_build_non_exhaustive_match_all_arms_guarded)]
pub(crate) struct NonExhaustiveMatchAllArmsGuarded;

#[derive(Diagnostic)]
#[diag(mir_build_static_in_pattern, code = E0158)]
pub(crate) struct StaticInPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(mir_build_static_in_pattern_def)]
    pub(crate) static_span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_param_in_pattern, code = E0158)]
pub(crate) struct ConstParamInPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(mir_build_const_param_in_pattern_def)]
    pub(crate) const_span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_non_const_path, code = E0080)]
pub(crate) struct NonConstPath {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unreachable_pattern)]
pub(crate) struct UnreachablePattern<'tcx> {
    #[label]
    pub(crate) span: Option<Span>,
    #[label(mir_build_unreachable_matches_no_values)]
    pub(crate) matches_no_values: Option<Span>,
    pub(crate) matches_no_values_ty: Ty<'tcx>,
    #[note(mir_build_unreachable_uninhabited_note)]
    pub(crate) uninhabited_note: Option<()>,
    #[label(mir_build_unreachable_covered_by_catchall)]
    pub(crate) covered_by_catchall: Option<Span>,
    #[subdiagnostic]
    pub(crate) wanted_constant: Option<WantedConstant>,
    #[note(mir_build_unreachable_pattern_const_reexport_accessible)]
    pub(crate) accessible_constant: Option<Span>,
    #[note(mir_build_unreachable_pattern_const_inaccessible)]
    pub(crate) inaccessible_constant: Option<Span>,
    #[note(mir_build_unreachable_pattern_let_binding)]
    pub(crate) pattern_let_binding: Option<Span>,
    #[label(mir_build_unreachable_covered_by_one)]
    pub(crate) covered_by_one: Option<Span>,
    #[note(mir_build_unreachable_covered_by_many)]
    pub(crate) covered_by_many: Option<MultiSpan>,
    pub(crate) covered_by_many_n_more_count: usize,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub(crate) suggest_remove: Option<Span>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    mir_build_unreachable_pattern_wanted_const,
    code = "{const_path}",
    applicability = "machine-applicable"
)]
pub(crate) struct WantedConstant {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) is_typo: bool,
    pub(crate) const_name: String,
    pub(crate) const_path: String,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_pattern_depends_on_generic_parameter, code = E0158)]
pub(crate) struct ConstPatternDependsOnGenericParameter {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_could_not_eval_const_pattern)]
pub(crate) struct CouldNotEvalConstPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_lower_range_bound_must_be_less_than_or_equal_to_upper, code = E0030)]
pub(crate) struct LowerRangeBoundMustBeLessThanOrEqualToUpper {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[note(mir_build_teach_note)]
    pub(crate) teach: bool,
}

#[derive(Diagnostic)]
#[diag(mir_build_literal_in_range_out_of_bounds)]
pub(crate) struct LiteralOutOfRange<'tcx> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) min: i128,
    pub(crate) max: u128,
}

#[derive(Diagnostic)]
#[diag(mir_build_lower_range_bound_must_be_less_than_upper, code = E0579)]
pub(crate) struct LowerRangeBoundMustBeLessThanUpper {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_leading_irrefutable_let_patterns)]
#[note]
#[help]
pub(crate) struct LeadingIrrefutableLetPatterns {
    pub(crate) count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_trailing_irrefutable_let_patterns)]
#[note]
#[help]
pub(crate) struct TrailingIrrefutableLetPatterns {
    pub(crate) count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_bindings_with_variant_name, code = E0170)]
pub(crate) struct BindingsWithVariantName {
    #[suggestion(code = "{ty_path}::{name}", applicability = "machine-applicable")]
    pub(crate) suggestion: Option<Span>,
    pub(crate) ty_path: String,
    pub(crate) name: Ident,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_if_let)]
#[note]
#[help]
pub(crate) struct IrrefutableLetPatternsIfLet {
    pub(crate) count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_if_let_guard)]
#[note]
#[help]
pub(crate) struct IrrefutableLetPatternsIfLetGuard {
    pub(crate) count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_let_else)]
#[note]
#[help]
pub(crate) struct IrrefutableLetPatternsLetElse {
    pub(crate) count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_while_let)]
#[note]
#[help]
pub(crate) struct IrrefutableLetPatternsWhileLet {
    pub(crate) count: usize,
}

#[derive(Diagnostic)]
#[diag(mir_build_borrow_of_moved_value)]
pub(crate) struct BorrowOfMovedValue<'tcx> {
    #[primary_span]
    #[label]
    #[label(mir_build_occurs_because_label)]
    pub(crate) binding_span: Span,
    #[label(mir_build_value_borrowed_label)]
    pub(crate) conflicts_ref: Vec<Span>,
    pub(crate) name: Ident,
    pub(crate) ty: Ty<'tcx>,
    #[suggestion(code = "ref ", applicability = "machine-applicable")]
    pub(crate) suggest_borrowing: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(mir_build_multiple_mut_borrows)]
pub(crate) struct MultipleMutBorrows {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_already_borrowed)]
pub(crate) struct AlreadyBorrowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_already_mut_borrowed)]
pub(crate) struct AlreadyMutBorrowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_moved_while_borrowed)]
pub(crate) struct MovedWhileBorrowed {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) occurrences: Vec<Conflict>,
}

#[derive(Subdiagnostic)]
pub(crate) enum Conflict {
    #[label(mir_build_mutable_borrow)]
    Mut {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
    #[label(mir_build_borrow)]
    Ref {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
    #[label(mir_build_moved)]
    Moved {
        #[primary_span]
        span: Span,
        name: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag(mir_build_union_pattern)]
pub(crate) struct UnionPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_type_not_structural)]
pub(crate) struct TypeNotStructural<'tcx> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(mir_build_type_not_structural_def)]
    pub(crate) ty_def_span: Span,
    pub(crate) ty: Ty<'tcx>,
    #[note(mir_build_type_not_structural_tip)]
    pub(crate) manual_partialeq_impl_span: Option<Span>,
    #[note(mir_build_type_not_structural_more_info)]
    pub(crate) manual_partialeq_impl_note: bool,
}

#[derive(Diagnostic)]
#[diag(mir_build_non_partial_eq_match)]
#[note(mir_build_type_not_structural_more_info)]
pub(crate) struct TypeNotPartialEq<'tcx> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(mir_build_invalid_pattern)]
pub(crate) struct InvalidPattern<'tcx> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) non_sm_ty: Ty<'tcx>,
    pub(crate) prefix: String,
}

#[derive(Diagnostic)]
#[diag(mir_build_unsized_pattern)]
pub(crate) struct UnsizedPattern<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) non_sm_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(mir_build_nan_pattern)]
#[note]
#[help]
pub(crate) struct NaNPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_pointer_pattern)]
#[note]
pub(crate) struct PointerPattern {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_non_empty_never_pattern)]
#[note]
pub(crate) struct NonEmptyNeverPattern<'tcx> {
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(mir_build_exceeds_mcdc_condition_limit)]
pub(crate) struct MCDCExceedsConditionLimit {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) num_conditions: usize,
    pub(crate) max_conditions: usize,
}

#[derive(Diagnostic)]
#[diag(mir_build_pattern_not_covered, code = E0005)]
pub(crate) struct PatternNotCovered<'s, 'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) origin: &'s str,
    #[subdiagnostic]
    pub(crate) uncovered: Uncovered,
    #[subdiagnostic]
    pub(crate) inform: Option<Inform>,
    #[label(mir_build_confused)]
    pub(crate) interpreted_as_const: Option<Span>,
    #[subdiagnostic]
    pub(crate) interpreted_as_const_sugg: Option<InterpretedAsConst>,
    #[subdiagnostic]
    pub(crate) adt_defined_here: Option<AdtDefinedHere<'tcx>>,
    #[note(mir_build_privately_uninhabited)]
    pub(crate) witness_1_is_privately_uninhabited: bool,
    #[note(mir_build_pattern_ty)]
    pub(crate) _p: (),
    pub(crate) pattern_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub(crate) let_suggestion: Option<SuggestLet>,
    #[subdiagnostic]
    pub(crate) misc_suggestion: Option<MiscPatternSuggestion>,
}

#[derive(Subdiagnostic)]
#[note(mir_build_inform_irrefutable)]
#[note(mir_build_more_information)]
pub(crate) struct Inform;

pub(crate) struct AdtDefinedHere<'tcx> {
    pub(crate) adt_def_span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) variants: Vec<Variant>,
}

pub(crate) struct Variant {
    pub(crate) span: Span,
}

impl<'tcx> Subdiagnostic for AdtDefinedHere<'tcx> {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("ty", self.ty);
        let mut spans = MultiSpan::from(self.adt_def_span);

        for Variant { span } in self.variants {
            spans.push_span_label(span, fluent::mir_build_variant_defined_here);
        }

        diag.span_note(spans, fluent::mir_build_adt_defined_here);
    }
}

#[derive(Subdiagnostic)]
#[suggestion(
    mir_build_interpreted_as_const,
    code = "{variable}_var",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct InterpretedAsConst {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) variable: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum SuggestLet {
    #[multipart_suggestion(mir_build_suggest_if_let, applicability = "has-placeholders")]
    If {
        #[suggestion_part(code = "if ")]
        start_span: Span,
        #[suggestion_part(code = " {{ todo!() }}")]
        semi_span: Span,
        count: usize,
    },
    #[suggestion(
        mir_build_suggest_let_else,
        code = " else {{ todo!() }}",
        applicability = "has-placeholders"
    )]
    Else {
        #[primary_span]
        end_span: Span,
        count: usize,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum MiscPatternSuggestion {
    #[suggestion(
        mir_build_suggest_attempted_int_lit,
        code = "_",
        applicability = "maybe-incorrect"
    )]
    AttemptedIntegerLiteral {
        #[primary_span]
        start_span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag(mir_build_rust_2024_incompatible_pat)]
pub(crate) struct Rust2024IncompatiblePat {
    #[subdiagnostic]
    pub(crate) sugg: Rust2024IncompatiblePatSugg,
    pub(crate) bad_modifiers: bool,
    pub(crate) bad_ref_pats: bool,
    pub(crate) is_hard_error: bool,
}

pub(crate) struct Rust2024IncompatiblePatSugg {
    /// If true, our suggestion is to elide explicit binding modifiers.
    /// If false, our suggestion is to make the pattern fully explicit.
    pub(crate) suggest_eliding_modes: bool,
    pub(crate) suggestion: Vec<(Span, String)>,
    pub(crate) ref_pattern_count: usize,
    pub(crate) binding_mode_count: usize,
    /// Labels for where incompatibility-causing by-ref default binding modes were introduced.
    pub(crate) default_mode_labels: FxIndexMap<Span, ty::Mutability>,
}

impl Subdiagnostic for Rust2024IncompatiblePatSugg {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        // Format and emit explanatory notes about default binding modes. Reversing the spans' order
        // means if we have nested spans, the innermost ones will be visited first.
        for (span, def_br_mutbl) in self.default_mode_labels.into_iter().rev() {
            // Don't point to a macro call site.
            if !span.from_expansion() {
                let note_msg = "matching on a reference type with a non-reference pattern changes the default binding mode";
                let label_msg =
                    format!("this matches on type `{}_`", def_br_mutbl.ref_prefix_str());
                let mut label = MultiSpan::from(span);
                label.push_span_label(span, label_msg);
                diag.span_note(label, note_msg);
            }
        }

        // Format and emit the suggestion.
        let applicability =
            if self.suggestion.iter().all(|(span, _)| span.can_be_used_for_suggestions()) {
                Applicability::MachineApplicable
            } else {
                Applicability::MaybeIncorrect
            };
        let msg = if self.suggest_eliding_modes {
            let plural_modes = pluralize!(self.binding_mode_count);
            format!("remove the unnecessary binding modifier{plural_modes}")
        } else {
            let plural_derefs = pluralize!(self.ref_pattern_count);
            let and_modes = if self.binding_mode_count > 0 {
                format!(" and variable binding mode{}", pluralize!(self.binding_mode_count))
            } else {
                String::new()
            };
            format!("make the implied reference pattern{plural_derefs}{and_modes} explicit")
        };
        // FIXME(dianne): for peace of mind, don't risk emitting a 0-part suggestion (that panics!)
        if !self.suggestion.is_empty() {
            diag.multipart_suggestion_verbose(msg, self.suggestion, applicability);
        }
    }
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_invalid_update)]
pub(crate) struct LoopMatchInvalidUpdate {
    #[primary_span]
    pub lhs: Span,
    #[label]
    pub scrutinee: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_invalid_match)]
#[note]
pub(crate) struct LoopMatchInvalidMatch {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_unsupported_type)]
#[note]
pub(crate) struct LoopMatchUnsupportedType<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_bad_statements)]
pub(crate) struct LoopMatchBadStatements {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_bad_rhs)]
pub(crate) struct LoopMatchBadRhs {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_missing_assignment)]
pub(crate) struct LoopMatchMissingAssignment {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_loop_match_arm_with_guard)]
pub(crate) struct LoopMatchArmWithGuard {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_continue_bad_const)]
pub(crate) struct ConstContinueBadConst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_continue_missing_value)]
pub(crate) struct ConstContinueMissingValue {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_continue_unknown_jump_target)]
#[note]
pub(crate) struct ConstContinueUnknownJumpTarget {
    #[primary_span]
    pub span: Span,
}
