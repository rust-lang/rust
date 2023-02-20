use crate::{
    fluent_generated as fluent,
    thir::pattern::{deconstruct_pat::{Constructor, DeconstructedPat}, MatchCheckCtxt},
};
use rustc_errors::Handler;
use rustc_errors::{
    error_code, AddToDiagnostic, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
    Handler, IntoDiagnostic, MultiSpan, SubdiagnosticMessage,
};
use rustc_hir::def::Res;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::thir::Pat;
use rustc_middle::ty::{self, AdtDef, Ty};
use rustc_span::{symbol::Ident, Span};

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
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_nameless, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeNameless {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
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
}

#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfInlineAssemblyRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_initializing_type_with_requires_unsafe, code = "E0133")]
#[note]
pub struct InitializingTypeWithRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
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
}

#[derive(Diagnostic)]
#[diag(mir_build_mutable_static_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfMutableStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_extern_static_requires_unsafe, code = "E0133")]
#[note]
pub struct UseOfExternStaticRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_deref_raw_pointer_requires_unsafe, code = "E0133")]
#[note]
pub struct DerefOfRawPointerRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_union_field_requires_unsafe, code = "E0133")]
#[note]
pub struct AccessToUnionFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(mir_build_mutation_of_layout_constrained_field_requires_unsafe, code = "E0133")]
#[note]
pub struct MutationOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
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
}

#[derive(Diagnostic)]
#[diag(mir_build_borrow_of_layout_constrained_field_requires_unsafe, code = "E0133")]
#[note]
pub struct BorrowOfLayoutConstrainedFieldRequiresUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
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
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe, code = "E0133")]
#[note]
pub struct CallToFunctionWithRequiresUnsafe<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
}

#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code = "E0133")]
#[note]
pub struct CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub function: &'a str,
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
    pub scrut_ty: Ty<'tcx>,
    pub type_note: TypeNote<'tcx>,
    pub ref_note: Option<RefNote>,
}

impl<'a> IntoDiagnostic<'a> for NonExhaustivePatternsTypeNotEmpty<'_, '_, '_> {
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_span_err_with_code(
            self.span,
            fluent::mir_build_non_exhaustive_patterns_type_not_empty,
            error_code!(E0004),
        );

        let peeled_ty = self.scrut_ty.peel_refs();
        diag.set_arg("scrut_ty", self.scrut_ty);
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

        match self.type_note {
            TypeNote::MarkedExhaustive { .. } => {
                diag.note(fluent::mir_build_type_note_non_exhaustive)
            }
            TypeNote::NotMarkedExhaustive { .. } => {
                diag.note(fluent::mir_build_type_note)
            }
        };

        if self.ref_note.is_some() {
            diag.note(fluent::mir_build_ref_note);
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
    #[suggestion(code = "{ty_path}::{ident}", applicability = "machine-applicable")]
    pub suggestion: Option<Span>,
    pub ty_path: String,
    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_irrefutable_let_patterns_generic_let)]
#[note]
#[help]
pub struct IrrefutableLetPatternsGenericLet {
    pub count: usize,
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
    pub span: Span,
    #[label]
    #[label(mir_build_occurs_because_label)]
    pub binding_span: Span,
    #[label(mir_build_value_borrowed_label)]
    pub conflicts_ref: Vec<Span>,
    pub name: Ident,
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
    pub occurences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_already_borrowed)]
pub struct AlreadyBorrowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_already_mut_borrowed)]
pub struct AlreadyMutBorrowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurences: Vec<Conflict>,
}

#[derive(Diagnostic)]
#[diag(mir_build_moved_while_borrowed)]
pub struct MovedWhileBorrowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub occurences: Vec<Conflict>,
}

#[derive(Subdiagnostic)]
pub enum Conflict {
    #[label(mir_build_mutable_borrow)]
    Mut {
        #[primary_span]
        span: Span,
        name: Ident,
    },
    #[label(mir_build_borrow)]
    Ref {
        #[primary_span]
        span: Span,
        name: Ident,
    },
    #[label(mir_build_moved)]
    Moved {
        #[primary_span]
        span: Span,
        name: Ident,
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
pub struct IndirectStructuralMatch<'tcx> {
    pub non_sm_ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag(mir_build_nontrivial_structural_match)]
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
        witnesses: &[DeconstructedPat<'p, 'tcx>],
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
    #[note(mir_build_pattern_ty)]
    pub _p: (),
    pub pattern_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub let_suggestion: Option<SuggestLet>,
    #[subdiagnostic]
    pub misc_suggestion: Option<MiscPatternSuggestion>,
    #[subdiagnostic]
    pub res_defined_here: Option<ResDefinedHere>,
}

#[derive(Subdiagnostic)]
#[note(mir_build_inform_irrefutable)]
#[note(mir_build_more_information)]
pub struct Inform;

pub(crate) struct AdtDefinedHere<'tcx> {
    adt_def_span: Span,
    ty: Ty<'tcx>,
    variants: Vec<Span>,
}

impl<'tcx> AdtDefinedHere<'tcx> {
    pub fn new<'p>(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        witnesses: &[DeconstructedPat<'p, 'tcx>],
    ) -> Option<Self> {
        fn maybe_point_at_variant<'a, 'p: 'a, 'tcx: 'a>(
            cx: &MatchCheckCtxt<'p, 'tcx>,
            def: AdtDef<'tcx>,
            patterns: impl Iterator<Item = &'a DeconstructedPat<'p, 'tcx>>,
        ) -> Vec<Span> {
            let mut covered = vec![];
            for pattern in patterns {
                if let Constructor::Variant(variant_index) = pattern.ctor() {
                    if let ty::Adt(this_def, _) = pattern.ty().kind() && this_def.did() != def.did() {
                        continue;
                    }
                    let sp = def.variant(*variant_index).ident(cx.tcx).span;
                    if covered.contains(&sp) {
                        // Don't point at variants that have already been covered due to other patterns to avoid
                        // visual clutter.
                        continue;
                    }
                    covered.push(sp);
                }
                covered.extend(maybe_point_at_variant(cx, def, pattern.iter_fields()));
            }
            covered
        }

        let ty = ty.peel_refs();
        let ty::Adt(def, _) = ty.kind() else { None? };
        let adt_def_span = cx.tcx.hir().get_if_local(def.did())?.ident()?.span;
        let mut variants = vec![];

        for span in maybe_point_at_variant(&cx, *def, witnesses.iter().take(5)) {
            variants.push(span);
        }
        Some(AdtDefinedHere { adt_def_span, ty, variants })
    }
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
#[label(mir_build_res_defined_here)]
pub struct ResDefinedHere {
    #[primary_span]
    pub def_span: Span,
    pub res: Res,
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
    pub article: &'static str,
    pub variable: String,
    pub res: Res,
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
#[diag(mir_build_non_exhaustive_pattern, code = "E0004")]
pub(crate) struct NonExhaustivePatterns<'tcx> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub uncovered: Uncovered<'tcx>,
    #[subdiagnostic]
    pub adt_defined_here: Option<AdtDefinedHere<'tcx>>,
    #[subdiagnostic]
    pub type_note: TypeNote<'tcx>,
    #[subdiagnostic]
    pub no_fixed_max_value: Option<NoFixedMaxValue<'tcx>>,
    #[subdiagnostic]
    pub ppsm: Option<SuggestPrecisePointerSizeMatching<'tcx>>,
    #[subdiagnostic]
    pub ref_note: Option<RefNote>,
    #[subdiagnostic]
    pub suggest_arms: ArmSuggestions<'tcx>,
}

#[derive(Subdiagnostic)]
pub enum TypeNote<'tcx> {
    #[note(mir_build_type_note)]
    NotMarkedExhaustive { scrut_ty: Ty<'tcx> },
    #[note(mir_build_type_note_non_exhaustive)]
    MarkedExhaustive { scrut_ty: Ty<'tcx> },
}

impl<'tcx> TypeNote<'tcx> {
    pub fn new(scrut_ty: Ty<'tcx>) -> Self {
        match scrut_ty.kind() {
            ty::Adt(def, _) if def.is_variant_list_non_exhaustive() && !def.did().is_local() => {
                TypeNote::MarkedExhaustive { scrut_ty }
            }
            _ => TypeNote::NotMarkedExhaustive { scrut_ty },
        }
    }
}

#[derive(Subdiagnostic)]
pub enum AddArmKind<'tcx> {
    #[help(mir_build_suggest_wildcard_arm)]
    Wildcard,
    #[help(mir_build_suggest_single_arm)]
    Single { pat: Pat<'tcx> },
    #[help(mir_build_suggest_multiple_arms)]
    Multiple,
}

#[derive(Subdiagnostic)]
#[note(mir_build_no_fixed_maximum_value)]
pub struct NoFixedMaxValue<'tcx> {
    pub scrut_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[help(mir_build_suggest_precise_pointer_size_matching)]
pub struct SuggestPrecisePointerSizeMatching<'tcx> {
    pub scrut_ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
#[note(mir_build_ref_note)]
pub struct RefNote;

pub enum ArmSuggestions<'tcx> {
    OneLiner {
        suggest_msg: AddArmKind<'tcx>,
        pattern: Pat<'tcx>,
        span: Span,
    },
    MultipleLines {
        span: Span,
        prefix: String,
        indentation: String,
        postfix: String,
        arm_suggestions: Vec<Pat<'tcx>>,
        suggest_msg: AddArmKind<'tcx>,
    },
    Help {
        suggest_msg: AddArmKind<'tcx>,
    },
}

impl<'tcx> AddToDiagnostic for ArmSuggestions<'tcx> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, f: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use std::fmt::Write;

        match self {
            ArmSuggestions::OneLiner { suggest_msg, span, pattern } => {
                let suggestion = format!(", {pattern} => {{ todo!() }}");
                let suggest_msg = match suggest_msg {
                    AddArmKind::Wildcard => rustc_errors::fluent::mir_build_suggest_wildcard_arm,
                    AddArmKind::Single { pat } => {
                        diag.set_arg("pat", pat);
                        rustc_errors::fluent::mir_build_suggest_single_arm
                    }
                    AddArmKind::Multiple => rustc_errors::fluent::mir_build_suggest_multiple_arms,
                };
                diag.span_suggestion_verbose(
                    span,
                    suggest_msg,
                    suggestion,
                    Applicability::HasPlaceholders,
                );
            }
            ArmSuggestions::MultipleLines {
                suggest_msg,
                span,
                prefix,
                indentation,
                postfix,
                arm_suggestions,
            } => {
                let suggest_msg = match suggest_msg {
                    AddArmKind::Wildcard => rustc_errors::fluent::mir_build_suggest_wildcard_arm,
                    AddArmKind::Single { pat } => {
                        diag.set_arg("pat", pat);
                        rustc_errors::fluent::mir_build_suggest_single_arm
                    }
                    AddArmKind::Multiple => rustc_errors::fluent::mir_build_suggest_multiple_arms,
                };

                let mut suggestion = String::new();

                // Set the correct position to start writing arms
                suggestion.push_str(&prefix);

                let (truncate_at, need_wildcard) = match arm_suggestions.len() {
                    // Avoid writing a wildcard for one remaining arm
                    4 => (4, false),
                    // Otherwise, limit it at 3 arms + wildcard
                    n @ 0..=3 => (n, false),
                    _ => (3, true),
                };

                for pattern in arm_suggestions.iter().take(truncate_at) {
                    writeln!(&mut suggestion, "{indentation}{pattern} => {{ todo!() }},").unwrap();
                }
                if need_wildcard {
                    writeln!(&mut suggestion, "{indentation}_ => {{ todo!() }},").unwrap();
                }
                suggestion.push_str(&postfix);

                diag.span_suggestion_verbose(
                    span,
                    suggest_msg,
                    suggestion,
                    Applicability::HasPlaceholders,
                );
            }
            ArmSuggestions::Help { suggest_msg } => suggest_msg.add_to_diagnostic_with(diag, f),
        }
    }
}
