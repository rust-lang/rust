use rustc_errors::MultiSpan;
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionSubdiagnostic)]
#[note(trait_selection::note_access_through_trait_impl)]
pub(crate) struct NoteAccessThroughTraitImpl {
    pub(crate) kind: &'static str,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum NoteObligation<'tcx> {
    #[note(trait_selection::note_obligation_assignment_lhs_sized)]
    AssignmentLhsSized,

    #[note(trait_selection::note_obligation_binding_obligation)]
    BindingObligation {
        #[primary_span]
        span: Span,
        item_name: String,
    },
    #[note(trait_selection::note_obligation_binding_obligation)]
    BindingObligationMultispan {
        #[primary_span]
        multispan: MultiSpan,
        item_name: String,
    },

    #[note(trait_selection::note_obligation_builtin_derived_obligation)]
    BuiltinDerivedObligation { ty: Ty<'tcx> },
    #[note(trait_selection::note_obligation_builtin_derived_obligation_closure)]
    BuiltinDerivedObligationClosure {
        #[primary_span]
        span: Span,
    },
    #[note(trait_selection::note_obligation_builtin_derived_obligation_generator)]
    BuiltinDerivedObligationGenerator {
        #[primary_span]
        span: Span,
        kind: String,
    },
    #[note(trait_selection::note_obligation_builtin_derived_obligation_generator_witness)]
    BuiltinDerivedObligationGeneratorWitness { captured_types: String },
    #[note(trait_selection::note_obligation_builtin_derived_obligation)]
    BuiltinDerivedObligationSpan {
        #[primary_span]
        span: Span,
        ty: Ty<'tcx>,
    },

    #[note(trait_selection::note_obligation_coercion)]
    Coercion { target: String },
    #[note(trait_selection::note_obligation_compare_impl_item_obligation)]
    CompareImplItemObligation {
        #[primary_span]
        assoc_span: MultiSpan,
        predicate: String,
        item_name: Symbol,
        kind: String,
    },
    #[note(trait_selection::note_obligation_const_pattern_structural)]
    ConstPatternStructural,
    #[note(trait_selection::note_obligation_const_sized)]
    ConstSized,

    #[note(trait_selection::note_obligation_field_sized_enum)]
    FieldSizedEnum,
    #[help(trait_selection::note_obligation_field_sized_help)]
    FieldSizedHelp,
    #[note(trait_selection::note_obligation_field_sized_struct)]
    FieldSizedStruct,
    #[note(trait_selection::note_obligation_field_sized_struct_last)]
    FieldSizedStructLast,
    #[suggestion(
        trait_selection::note_obligation_field_sized_suggest_borrowed,
        code = "&",
        applicability = "machine-applicable"
    )]
    FieldSizedSuggestBorrowed {
        #[primary_span]
        span: Span,
    },
    #[note(trait_selection::note_obligation_field_sized_union)]
    FieldSizedUnion,

    #[label(trait_selection::note_obligation_function_argument_obligation_required_by)]
    FunctionArgumentObligationRequiredBy {
        #[primary_span]
        span: Span,
    },
    #[label(trait_selection::note_obligation_function_argument_obligation_tail_expr_type)]
    FunctionArgumentObligationTailExprType {
        #[primary_span]
        span: Span,
        // TODO: fmt::Debug?
        ty: Ty<'tcx>,
    },

    #[note(trait_selection::note_obligation_impl_derived_obligation)]
    ImplDerivedObligation { ty: Ty<'tcx>, trait_path: String },
    #[note(trait_selection::note_obligation_impl_derived_obligation_redundant_hidden)]
    ImplDerivedObligationRedundantHidden { count: usize },
    #[note(trait_selection::note_obligation_impl_derived_obligation)]
    ImplDerivedObligationSpan {
        #[primary_span]
        span: Span,
        ty: Ty<'tcx>,
        trait_path: String,
    },
    #[note(trait_selection::note_obligation_impl_derived_obligation)]
    ImplDerivedObligationSpans {
        #[primary_span]
        spans: MultiSpan,
        ty: Ty<'tcx>,
        trait_path: String,
    },

    #[note(trait_selection::note_obligation_inline_asm_sized)]
    InlineAsmSized,

    #[note(trait_selection::note_obligation_object_cast_obligation)]
    ObjectCastObligation { concrete_ty: String, object_ty: String },
    #[note(trait_selection::note_obligation_object_type_bound)]
    ObjectTypeBound { object_ty: Ty<'tcx>, region: String },
    #[label(trait_selection::note_obligation_opaque_return_type_label)]
    OpaqueReturnTypeLabel {
        #[primary_span]
        expr_span: Span,
        expr_ty: Ty<'tcx>,
    },
    #[note(trait_selection::note_obligation_projection_wf)]
    ProjectionWf { data: String },
    #[note(trait_selection::note_obligation_reference_outlives_referent)]
    ReferenceOutlivesReferent { ref_ty: Ty<'tcx> },

    #[note(trait_selection::note_obligation_repeat_element_copy)]
    RepeatElementCopy,
    #[help(trait_selection::note_obligation_repeat_element_copy_help_const_fn)]
    RepeatElementCopyHelpConstFn { example_a: &'static str, example_b: &'static str },
    #[help(trait_selection::note_obligation_repeat_element_copy_help_nightly_const_fn)]
    RepeatElementCopyHelpNightlyConstFn,

    #[note(trait_selection::note_obligation_shared_static)]
    SharedStatic,

    #[note(trait_selection::note_obligation_sized_argument_type)]
    SizedArgumentType,
    #[help(trait_selection::note_obligation_sized_argument_type_help_nightly_unsized_fn_params)]
    SizedArgumentTypeHelpNightlyUnsizedFnParams,
    #[suggestion_verbose(
        trait_selection::note_obligation_sized_argument_type_suggest_borrowed,
        code = "&",
        applicability = "machine-applicable"
    )]
    SizedArgumentTypeSuggestBorrowed {
        #[primary_span]
        span: Span,
    },

    #[note(trait_selection::note_obligation_sized_box_type)]
    SizedBoxType,
    #[note(trait_selection::note_obligation_sized_return_type)]
    SizedReturnType,
    #[note(trait_selection::note_obligation_sized_yield_type)]
    SizedYieldType,
    #[note(trait_selection::note_obligation_slice_or_array_elem)]
    SliceOrArrayElem,
    #[note(trait_selection::note_obligation_struct_initializer_sized)]
    StructInitializerSized,

    #[help(trait_selection::note_obligation_trivial_bound_help)]
    TrivialBoundHelp,
    #[help(trait_selection::note_obligation_trivial_bound_help_nightly)]
    TrivialBoundHelpNightly,

    #[note(trait_selection::note_obligation_tuple_elem)]
    TupleElem,
    #[note(trait_selection::note_obligation_tuple_initializer_sized)]
    TupleInitializerSized,

    #[help(trait_selection::note_obligation_variable_type_help_unsized_locals)]
    VariableTypeHelpUnsizedLocals,
    #[note(trait_selection::note_obligation_variable_type_local)]
    VariableTypeLocal,
    #[suggestion_verbose(
        trait_selection::note_obligation_variable_type_local_expression,
        code = "&",
        applicability = "machine-applicable"
    )]
    VariableTypeLocalExpression {
        #[primary_span]
        span: Span,
    },
    #[suggestion_verbose(
        trait_selection::note_obligation_variable_type_param,
        code = "&",
        applicability = "machine-applicable"
    )]
    VariableTypeParam {
        #[primary_span]
        span: Span,
    },
}

#[derive(SessionSubdiagnostic)]
#[label(trait_selection::point_at_returns_when_relevant)]
pub(crate) struct PointAtReturnsWhenRelevant<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum SuggestAddReferenceToArg<'tcx> {
    #[label(trait_selection::suggest_add_reference_to_arg_label)]
    Label {
        #[primary_span]
        span: Span,
        trait_path: String,
        ty: Ty<'tcx>,
    },
    #[note(trait_selection::suggest_add_reference_to_arg_note)]
    Note { trait_bound: String },
    #[suggestion_verbose(
        trait_selection::suggest_add_reference_to_arg,
        code = "&{mutability}",
        applicability = "maybe-incorrect"
    )]
    Suggest {
        #[primary_span]
        span: Span,
        is_mut: &'static str,
        mutability: &'static str,
    },
}

#[derive(SessionSubdiagnostic)]
#[suggestion_verbose(
    trait_selection::suggest_await_before_try,
    code = ".await",
    applicability = "maybe-incorrect"
)]
pub(crate) struct SuggestAwaitBeforeTry {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(
    trait_selection::suggest_borrowing_for_object_cast,
    code = "&",
    applicability = "maybe-incorrect"
)]
pub(crate) struct SuggestBorrowingForObjectCast<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) self_ty: Ty<'tcx>,
    pub(crate) object_ty: Ty<'tcx>,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum SuggestChangeMut<'tcx> {
    #[suggestion(
        trait_selection::suggest_change_borrow_mutability,
        code = "&mut ",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
    },
    #[note(trait_selection::note_implemented_for_other)]
    Note {
        trait_path: String,
        // TODO: fmt::Debug?
        suggested_ty: Ty<'tcx>,
        // TODO: fmt::Debug?
        original_ty: Ty<'tcx>,
    },
}

#[derive(SessionSubdiagnostic)]
#[suggestion_verbose(
    trait_selection::suggest_dereferencing_index,
    code = "*",
    applicability = "machine-applicable"
)]
pub(crate) struct SuggestDereferencingIndex {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion_verbose(
    trait_selection::suggest_derive,
    code = "{annotation}\n",
    applicability = "maybe-incorrect"
)]
pub(crate) struct SuggestDerive<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) self_ty: Ty<'tcx>,
    pub(crate) annotation: String,
}

#[derive(SessionSubdiagnostic)]
#[suggestion_verbose(
    trait_selection::suggest_floating_point_literal,
    code = ".0",
    applicability = "maybe-incorrect"
)]
pub(crate) struct SuggestFloatingPointLiteral {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum SuggestFnCall {
    #[label(trait_selection::suggest_fn_call_closure)]
    Closure {
        #[primary_span]
        span: Span,
    },
    #[label(trait_selection::suggest_fn_call_fn)]
    Fn {
        #[primary_span]
        span: Span,
    },
    #[help(trait_selection::suggest_fn_call_help)]
    Help { callable: &'static str, snippet: String },
    #[note(trait_selection::suggest_fn_call_msg)]
    Note { callable: &'static str },
    #[suggestion(
        trait_selection::suggest_fn_call_msg,
        code = "{sugg}",
        applicability = "has-placeholders"
    )]
    Suggest {
        #[primary_span]
        span: Span,
        callable: &'static str,
        sugg: String,
    },
}

#[derive(SessionSubdiagnostic)]
#[suggestion(
    trait_selection::suggest_fully_qualified_path,
    code = "<Type as {trait_str}>::{assoc_item}",
    applicability = "has-placeholders"
)]
pub(crate) struct SuggestFullyQualifiedPath {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) trait_str: String,
    pub(crate) assoc_item: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(trait_selection::suggest_impl_trait, code = "E0746")]
pub(crate) struct SuggestImplTrait {
    pub(crate) trait_obj: String,
    pub(crate) last_ty: String,

    #[suggestion(code = "impl {trait_obj}", applicability = "machine-applicable")]
    pub(crate) use_impl_trait: Option<Span>,
    #[note(trait_selection::could_return_if_object_safe)]
    pub(crate) could_return_if_object_safe: Option<()>,
    #[note(trait_selection::trait_obj_msg)]
    pub(crate) trait_obj_msg: Option<()>,
    #[note(trait_selection::ret_vals_same_type)]
    pub(crate) ret_vals_same_type: Option<()>,
    #[note(trait_selection::impl_trait_msg)]
    pub(crate) impl_trait_msg: Option<()>,
    #[note(trait_selection::create_enum)]
    pub(crate) create_enum: Option<()>,
}

impl SuggestImplTrait {
    pub(crate) fn only_never_return(trait_obj: &str) -> Self {
        SuggestImplTrait {
            trait_obj: trait_obj.to_owned(),
            last_ty: String::new(),
            use_impl_trait: None,
            could_return_if_object_safe: None,
            trait_obj_msg: None,
            ret_vals_same_type: None,
            impl_trait_msg: None,
            create_enum: None,
        }
    }

    pub(crate) fn all_returns_have_same_type<'tcx>(
        span: Span,
        trait_obj: &str,
        last_ty: Ty<'tcx>,
    ) -> Self {
        SuggestImplTrait {
            trait_obj: trait_obj.to_owned(),
            last_ty: last_ty.to_string(),
            use_impl_trait: Some(span),
            could_return_if_object_safe: None,
            trait_obj_msg: None,
            ret_vals_same_type: None,
            impl_trait_msg: Some(()),
            create_enum: None,
        }
    }

    pub(crate) fn object_safe(trait_obj: &str) -> Self {
        SuggestImplTrait {
            trait_obj: trait_obj.to_owned(),
            last_ty: String::new(),
            use_impl_trait: None,
            could_return_if_object_safe: None,
            trait_obj_msg: Some(()),
            ret_vals_same_type: Some(()),
            impl_trait_msg: Some(()),
            create_enum: Some(()),
        }
    }

    pub(crate) fn not_object_safe(trait_obj: &str) -> Self {
        SuggestImplTrait {
            trait_obj: trait_obj.to_owned(),
            last_ty: String::new(),
            use_impl_trait: None,
            could_return_if_object_safe: Some(()),
            trait_obj_msg: Some(()),
            ret_vals_same_type: Some(()),
            impl_trait_msg: Some(()),
            create_enum: Some(()),
        }
    }
}

#[derive(SessionSubdiagnostic)]
#[help(trait_selection::suggest_new_overflow_limit)]
pub(crate) struct SuggestNewOverflowLimit {
    pub(crate) limit_attribute: String,
    pub(crate) crate_name: Symbol,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum SuggestRemoveAwait<'tcx> {
    #[label(trait_selection::suggest_remove_await_label_return)]
    LabelReturn {
        #[primary_span]
        span: Span,
        self_ty: Ty<'tcx>,
    },
    #[suggestion_verbose(
        trait_selection::suggest_remove_await,
        code = "",
        applicability = "machine-applicable"
    )]
    Suggest {
        #[primary_span]
        span: Span,
    },
    #[suggestion_verbose(
        trait_selection::suggest_remove_await_suggest_async,
        code = "{suggestion}",
        applicability = "maybe-incorrect"
    )]
    SuggestAsync {
        #[primary_span]
        span: Span,
        ident: Ident,
        suggestion: &'static str,
    },
}

#[derive(SessionSubdiagnostic)]
#[suggestion_short(
    trait_selection::suggest_remove_reference,
    code = "",
    applicability = "machine-applicable"
)]
pub(crate) struct SuggestRemoveReference {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) remove_refs: usize,
}

#[derive(SessionSubdiagnostic)]
#[label(trait_selection::suggest_semicolon_removal_label)]
pub(crate) struct SuggestSemicolonRemovalLabel<'tcx> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ty: Ty<'tcx>,
    pub(crate) trait_path: String,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(
    trait_selection::suggest_semicolon_removal,
    code = "",
    applicability = "machine-applicable"
)]
pub(crate) struct SuggestSemicolonRemoval {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(trait_selection::suggestions_type_mismatch_in_args, code = "E0631")]
pub(crate) struct TypeMismatchInArgs {
    pub(crate) argument_kind: &'static str,
    #[primary_span]
    #[label]
    pub(crate) span: Span,
    #[label(trait_selection::found_label)]
    pub(crate) found_span: Span,
}
