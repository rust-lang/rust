use crate::{
    fluent_generated as fluent,
    thir::pattern::{deconstruct_pat::DeconstructedPat, MatchCheckCtxt},
};
use rustc_errors::{
    error_code, AddToDiagnostic, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
    Handler, IntoDiagnostic, MultiSpan, SubdiagnosticMessage,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::thir::Pat;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

#[derive(LintDiagnostic)]
#[diag(mir_build_unconditional_recursion)]
#[help]
pub struct UnconditionalRecursion {
    #[label]
    pub span: Span,
    #[label(mir_build_unconditional_recursion_call_site_label)]
    pub call_sites: Vec<Span>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe<'a> {
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless)]
#[note]
pub struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_extern_static_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_union_field_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe)]
pub struct UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[label]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe)]
#[note]
pub struct UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe<'a> {
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafe<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_nameless, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeNameless {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfInlineAssemblyRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_initializing_type_with_requires_unsafe, code = "E0133")]
#[note]
pub struct InitializingTypeWithRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutable_static_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfMutableStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_extern_static_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfExternStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_deref_raw_pointer_requires_unsafe, code = "E0133")]
#[note]
pub struct DerefOfRawPointerRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_union_field_requires_unsafe, code = "E0133")]
#[note]
pub struct AccessToUnionFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutation_of_layout_constrained_field_requires_unsafe, code = "E0133")]
#[note]
pub struct MutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_borrow_of_layout_constrained_field_requires_unsafe, code = "E0133")]
#[note]
pub struct BorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(
    mir_build_borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,
    code = "E0133"
)]
#[note]
pub struct BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe, code = "E0133")]
#[note]
pub struct CallToFunctionWithRequiresUnsafe<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
    #[subdiagnostic]
    pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,
}

#[derive(Subdiagnostic)]
#[label(mir_build_unsafe_not_inherited)]
pub struct UnsafeNotInheritedNote {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unused_unsafe)]
pub struct UnusedUnsafe {
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub enclosing: Option<UnusedUnsafeEnclosing>,
}

#[derive(Subdiagnostic)]
pub enum UnusedUnsafeEnclosing {
    #[label(mir_build_unused_unsafe_enclosing_block_label)]
    Block {
        #[primary_span]
        span: Span,
    },
    #[label(mir_build_unused_unsafe_enclosing_fn_label)]
    Function {
        #[primary_span]
        span: Span,
    },
}

pub(crate) struct NonExhaustivePatternsTypeNotEmpty<'p, 'tcx, 'm> {
    pub cx: &'m MatchCheckCtxt<'p, 'tcx>,
    pub expr_span: Span,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

impl<'a> IntoDiagnostic<'a> for NonExhaustivePatternsTypeNotEmpty<'_, '_, '_> {
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            self.span,
            fluent::mir_build_non_exhaustive_patterns_type_not_empty,
            error_code!(E0004),
        );

        let peeled_ty = self.ty.peel_refs();
        diag.set_arg("ty", self.ty);
        diag.set_arg("peeled_ty", peeled_ty);

        if let ty::Adt(def, _) = peeled_ty.kind() {
            let def_span = self
                .cx
                .tcx
                .hir()
                .get_if_local(def.did())
                .and_then(|node| node.ident())
                .map(|ident| ident.span)
                .unwrap_or_else(|| self.cx.tcx.def_span(def.did()));

            // workaround to make test pass
            let mut span: MultiSpan = def_span.into();
            span.push_span_label(def_span, "");

            diag.span_note(span, fluent::mir_build_def_note);
        }

        let is_variant_list_non_exhaustive = matches!(self.ty.kind(),
            ty::Adt(def, _) if def.is_variant_list_non_exhaustive() && !def.did().is_local());
        if is_variant_list_non_exhaustive {
            diag.note(fluent::mir_build_non_exhaustive_type_note);
        } else {
            diag.note(fluent::mir_build_type_note);
        }

        if let ty::Ref(_, sub_ty, _) = self.ty.kind() {
            if !sub_ty.is_inhabited_from(self.cx.tcx, self.cx.module, self.cx.param_env) {
                diag.note(fluent::mir_build_reference_note);
            }
        }

        let mut suggestion = None;
        let sm = self.cx.tcx.sess.source_map();
        if self.span.eq_ctxt(self.expr_span) {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(self.span) {
                (format!("\n{}", snippet), "    ")
            } else {
                (" ".to_string(), "")
            };
            suggestion = Some((
                self.span.shrink_to_hi().with_hi(self.expr_span.hi()),
                format!(
                    " {{{indentation}{more}_ => todo!(),{indentation}}}",
                    indentation = indentation,
                    more = more,
                ),
            ));
        }

        if let Some((span, sugg)) = suggestion {
            diag.span_suggestion_verbose(
                span,
                fluent::mir_build_suggestion,
                sugg,
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
pub struct NonExhaustiveMatchAllArmsGuarded;

#[derive(Diagnostic)]
#[diag(mir_build_static_in_pattern, code = "E0158")]
pub struct StaticInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_assoc_const_in_pattern, code = "E0158")]
pub struct AssocConstInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_param_in_pattern, code = "E0158")]
pub struct ConstParamInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_non_const_path, code = "E0080")]
pub struct NonConstPath {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_unreachable_pattern)]
pub struct UnreachablePattern {
    #[label]
    pub span: Option<Span>,
    #[label(mir_build_catchall_label)]
    pub catchall: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(mir_build_const_pattern_depends_on_generic_parameter)]
pub struct ConstPatternDependsOnGenericParameter {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_could_not_eval_const_pattern)]
pub struct CouldNotEvalConstPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_lower_range_bound_must_be_less_than_or_equal_to_upper, code = "E0030")]
pub struct LowerRangeBoundMustBeLessThanOrEqualToUpper {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note(mir_build_teach_note)]
    pub teach: Option<()>,
}

#[derive(Diagnostic)]
#[diag(mir_build_literal_in_range_out_of_bounds)]
pub struct LiteralOutOfRange<'tcx> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub max: u128,
}

#[derive(Diagnostic)]
#[diag(mir_build_lower_range_bound_must_be_less_than_upper, code = "E0579")]
pub struct LowerRangeBoundMustBeLessThanUpper {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_leading_irrefutable_let_patterns)]
#[note]
#[help]
pub struct LeadingIrrefutableLetPatterns {
    pub count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_trailing_irrefutable_let_patterns)]
#[note]
#[help]
pub struct TrailingIrrefutableLetPatterns {
    pub count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_bindings_with_variant_name, code = "E0170")]
pub struct BindingsWithVariantName {
    #[suggestion(code = "{ty_path}::{name}", applicability = "machine-applicable")]
    pub suggestion: Option<Span>,
    pub ty_path: String,
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_if_let)]
#[note]
#[help]
pub struct IrrefutableLetPatternsIfLet {
    pub count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_if_let_guard)]
#[note]
#[help]
pub struct IrrefutableLetPatternsIfLetGuard {
    pub count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_let_else)]
#[note]
#[help]
pub struct IrrefutableLetPatternsLetElse {
    pub count: usize,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_while_let)]
#[note]
#[help]
pub struct IrrefutableLetPatternsWhileLet {
    pub count: usize,
}

#[derive(Diagnostic)]
#[diag(mir_build_borrow_of_moved_value)]
pub struct BorrowOfMovedValue<'tcx> {
    #[primary_span]
    #[label]
    #[label(mir_build_occurs_because_label)]
    pub binding_span: Span,
    #[label(mir_build_value_borrowed_label)]
    pub conflicts_ref: Vec<Span>,
    pub name: Symbol,
    pub ty: Ty<'tcx>,
    #[suggestion(code = "ref ", applicability = "machine-applicable")]
    pub suggest_borrowing: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(mir_build_multiple_mut_borrows)]
pub struct MultipleMutBorrows {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_already_borrowed)]
pub struct AlreadyBorrowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_already_mut_borrowed)]
pub struct AlreadyMutBorrowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurrences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_moved_while_borrowed)]
pub struct MovedWhileBorrowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurrences: Vec<Conflict>,
}

#[derive(Subdiagnostic)]
pub enum Conflict {
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
pub struct UnionPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_type_not_structural)]
#[note(mir_build_type_not_structural_tip)]
#[note(mir_build_type_not_structural_more_info)]
pub struct TypeNotStructural<'tcx> {
    #[primary_span]
    pub span: Span,
    pub non_sm_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(mir_build_invalid_pattern)]
pub struct InvalidPattern<'tcx> {
    #[primary_span]
    pub span: Span,
    pub non_sm_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(mir_build_unsized_pattern)]
pub struct UnsizedPattern<'tcx> {
    #[primary_span]
    pub span: Span,
    pub non_sm_ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_float_pattern)]
pub struct FloatPattern;

#[derive(LintDiagnostic)]
#[diag(mir_build_pointer_pattern)]
pub struct PointerPattern;

#[derive(LintDiagnostic)]
#[diag(mir_build_indirect_structural_match)]
#[note(mir_build_type_not_structural_tip)]
#[note(mir_build_type_not_structural_more_info)]
pub struct IndirectStructuralMatch<'tcx> {
    pub non_sm_ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_nontrivial_structural_match)]
#[note(mir_build_type_not_structural_tip)]
#[note(mir_build_type_not_structural_more_info)]
pub struct NontrivialStructuralMatch<'tcx> {
    pub non_sm_ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_overlapping_range_endpoints)]
#[note]
pub struct OverlappingRangeEndpoints<'tcx> {
    #[label(mir_build_range)]
    pub range: Span,
    #[subdiagnostic]
    pub overlap: Vec<Overlap<'tcx>>,
}

pub struct Overlap<'tcx> {
    pub span: Span,
    pub range: Pat<'tcx>,
}

impl<'tcx> AddToDiagnostic for Overlap<'tcx> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        let Overlap { span, range } = self;

        // FIXME(mejrs) unfortunately `#[derive(LintDiagnostic)]`
        // does not support `#[subdiagnostic(eager)]`...
        let message = format!("this range overlaps on `{range}`...");
        diag.span_label(span, message);
    }
}

#[derive(LintDiagnostic)]
#[diag(mir_build_non_exhaustive_omitted_pattern)]
#[help]
#[note]
pub(crate) struct NonExhaustiveOmittedPattern<'tcx> {
    pub scrut_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub uncovered: Uncovered<'tcx>,
}

#[derive(Subdiagnostic)]
#[label(mir_build_uncovered)]
pub(crate) struct Uncovered<'tcx> {
    #[primary_span]
    span: Span,
    count: usize,
    witness_1: Pat<'tcx>,
    witness_2: Pat<'tcx>,
    witness_3: Pat<'tcx>,
    remainder: usize,
}

impl<'tcx> Uncovered<'tcx> {
    pub fn new<'p>(
        span: Span,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        witnesses: Vec<DeconstructedPat<'p, 'tcx>>,
    ) -> Self {
        let witness_1 = witnesses.get(0).unwrap().to_pat(cx);
        Self {
            span,
            count: witnesses.len(),
            // Substitute dummy values if witnesses is smaller than 3. These will never be read.
            witness_2: witnesses.get(1).map(|w| w.to_pat(cx)).unwrap_or_else(|| witness_1.clone()),
            witness_3: witnesses.get(2).map(|w| w.to_pat(cx)).unwrap_or_else(|| witness_1.clone()),
            witness_1,
            remainder: witnesses.len().saturating_sub(3),
        }
    }
}

#[derive(Diagnostic)]
#[diag(mir_build_pattern_not_covered, code = "E0005")]
pub(crate) struct PatternNotCovered<'s, 'tcx> {
    #[primary_span]
    pub span: Span,
    pub origin: &'s str,
    #[subdiagnostic]
    pub uncovered: Uncovered<'tcx>,
    #[subdiagnostic]
    pub inform: Option<Inform>,
    #[subdiagnostic]
    pub interpreted_as_const: Option<InterpretedAsConst>,
    #[subdiagnostic]
    pub adt_defined_here: Option<AdtDefinedHere<'tcx>>,
    #[note(mir_build_privately_uninhabited)]
    pub witness_1_is_privately_uninhabited: Option<()>,
    #[note(mir_build_pattern_ty)]
    pub _p: (),
    pub pattern_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub let_suggestion: Option<SuggestLet>,
    #[subdiagnostic]
    pub misc_suggestion: Option<MiscPatternSuggestion>,
}

#[derive(Subdiagnostic)]
#[note(mir_build_inform_irrefutable)]
#[note(mir_build_more_information)]
pub struct Inform;

pub struct AdtDefinedHere<'tcx> {
    pub adt_def_span: Span,
    pub ty: Ty<'tcx>,
    pub variants: Vec<Variant>,
}

pub struct Variant {
    pub span: Span,
}

impl<'tcx> AddToDiagnostic for AdtDefinedHere<'tcx> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        diag.set_arg("ty", self.ty);
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
    applicability = "maybe-incorrect"
)]
#[label(mir_build_confused)]
pub struct InterpretedAsConst {
    #[primary_span]
    pub span: Span,
    pub variable: String,
}

#[derive(Subdiagnostic)]
pub enum SuggestLet {
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
pub enum MiscPatternSuggestion {
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

#[derive(Diagnostic)]
#[diag(mir_build_rustc_box_attribute_error)]
pub struct RustcBoxAttributeError {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub reason: RustcBoxAttrReason,
}

#[derive(Subdiagnostic)]
pub enum RustcBoxAttrReason {
    #[note(mir_build_attributes)]
    Attributes,
    #[note(mir_build_not_box)]
    NotBoxNew,
    #[note(mir_build_missing_box)]
    MissingBox,
}
