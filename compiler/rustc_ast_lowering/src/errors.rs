use rustc_errors::DiagArgFromDisplay;
use rustc_errors::codes::*;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag(ast_lowering_generic_type_with_parentheses, code = E0214)]
pub(crate) struct GenericTypeWithParentheses {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<UseAngleBrackets>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(ast_lowering_use_angle_brackets, applicability = "maybe-incorrect")]
pub(crate) struct UseAngleBrackets {
    #[suggestion_part(code = "<")]
    pub open_param: Span,
    #[suggestion_part(code = ">")]
    pub close_param: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_abi, code = E0703)]
#[note]
pub(crate) struct InvalidAbi {
    #[primary_span]
    #[label]
    pub span: Span,
    pub abi: Symbol,
    pub command: String,
    #[subdiagnostic]
    pub suggestion: Option<InvalidAbiSuggestion>,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_default_field_in_tuple)]
pub(crate) struct TupleStructWithDefault {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    ast_lowering_invalid_abi_suggestion,
    code = "\"{suggestion}\"",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct InvalidAbiSuggestion {
    #[primary_span]
    pub span: Span,
    pub suggestion: String,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_assoc_ty_parentheses)]
pub(crate) struct AssocTyParentheses {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: AssocTyParenthesesSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum AssocTyParenthesesSub {
    #[multipart_suggestion(ast_lowering_remove_parentheses)]
    Empty {
        #[suggestion_part(code = "")]
        parentheses_span: Span,
    },
    #[multipart_suggestion(ast_lowering_use_angle_brackets)]
    NotEmpty {
        #[suggestion_part(code = "<")]
        open_param: Span,
        #[suggestion_part(code = ">")]
        close_param: Span,
    },
}

#[derive(Diagnostic)]
#[diag(ast_lowering_misplaced_impl_trait, code = E0562)]
#[note]
pub(crate) struct MisplacedImplTrait<'a> {
    #[primary_span]
    pub span: Span,
    pub position: DiagArgFromDisplay<'a>,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_assoc_ty_binding_in_dyn)]
pub(crate) struct MisplacedAssocTyBinding {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " = impl", applicability = "maybe-incorrect", style = "verbose")]
    pub suggestion: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_underscore_expr_lhs_assign)]
pub(crate) struct UnderscoreExprLhsAssign {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_await_only_in_async_fn_and_blocks, code = E0728)]
pub(crate) struct AwaitOnlyInAsyncFnAndBlocks {
    #[primary_span]
    #[label]
    pub await_kw_span: Span,
    #[label(ast_lowering_this_not_async)]
    pub item_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_coroutine_too_many_parameters, code = E0628)]
pub(crate) struct CoroutineTooManyParameters {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_closure_cannot_be_static, code = E0697)]
pub(crate) struct ClosureCannotBeStatic {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_functional_record_update_destructuring_assignment)]
pub(crate) struct FunctionalRecordUpdateDestructuringAssignment {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_async_coroutines_not_supported, code = E0727)]
pub(crate) struct AsyncCoroutinesNotSupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_inline_asm_unsupported_target, code = E0472)]
pub(crate) struct InlineAsmUnsupportedTarget {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_att_syntax_only_x86)]
pub(crate) struct AttSyntaxOnlyX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_abi_specified_multiple_times)]
pub(crate) struct AbiSpecifiedMultipleTimes {
    #[primary_span]
    pub abi_span: Span,
    pub prev_name: Symbol,
    #[label]
    pub prev_span: Span,
    #[note]
    pub equivalent: bool,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_clobber_abi_not_supported)]
pub(crate) struct ClobberAbiNotSupported {
    #[primary_span]
    pub abi_span: Span,
}

#[derive(Diagnostic)]
#[note]
#[diag(ast_lowering_invalid_abi_clobber_abi)]
pub(crate) struct InvalidAbiClobberAbi {
    #[primary_span]
    pub abi_span: Span,
    pub supported_abis: String,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_register)]
pub(crate) struct InvalidRegister<'a> {
    #[primary_span]
    pub op_span: Span,
    pub reg: Symbol,
    pub error: &'a str,
}

#[derive(Diagnostic)]
#[note]
#[diag(ast_lowering_invalid_register_class)]
pub(crate) struct InvalidRegisterClass {
    #[primary_span]
    pub op_span: Span,
    pub reg_class: Symbol,
    pub supported_register_classes: String,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_asm_template_modifier_reg_class)]
pub(crate) struct InvalidAsmTemplateModifierRegClass {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
    #[subdiagnostic]
    pub sub: InvalidAsmTemplateModifierRegClassSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidAsmTemplateModifierRegClassSub {
    #[note(ast_lowering_support_modifiers)]
    SupportModifier { class_name: Symbol, modifiers: String },
    #[note(ast_lowering_does_not_support_modifiers)]
    DoesNotSupportModifier { class_name: Symbol },
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_asm_template_modifier_const)]
pub(crate) struct InvalidAsmTemplateModifierConst {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_asm_template_modifier_sym)]
pub(crate) struct InvalidAsmTemplateModifierSym {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_asm_template_modifier_label)]
pub(crate) struct InvalidAsmTemplateModifierLabel {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_register_class_only_clobber)]
pub(crate) struct RegisterClassOnlyClobber {
    #[primary_span]
    pub op_span: Span,
    pub reg_class_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_register_class_only_clobber_stable)]
pub(crate) struct RegisterClassOnlyClobberStable {
    #[primary_span]
    pub op_span: Span,
    pub reg_class_name: Symbol,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_register_conflict)]
pub(crate) struct RegisterConflict<'a> {
    #[primary_span]
    #[label(ast_lowering_register1)]
    pub op_span1: Span,
    #[label(ast_lowering_register2)]
    pub op_span2: Span,
    pub reg1_name: &'a str,
    pub reg2_name: &'a str,
    #[help]
    pub in_out: Option<Span>,
}

#[derive(Diagnostic)]
#[help]
#[diag(ast_lowering_sub_tuple_binding)]
pub(crate) struct SubTupleBinding<'a> {
    #[primary_span]
    #[label]
    #[suggestion(
        ast_lowering_sub_tuple_binding_suggestion,
        style = "verbose",
        code = "..",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
    pub ident: Ident,
    pub ident_name: Symbol,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_extra_double_dot)]
pub(crate) struct ExtraDoubleDot<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(ast_lowering_previously_used_here)]
    pub prev_span: Span,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[note]
#[diag(ast_lowering_misplaced_double_dot)]
pub(crate) struct MisplacedDoubleDot {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_misplaced_relax_trait_bound)]
pub(crate) struct MisplacedRelaxTraitBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_match_arm_with_no_body)]
pub(crate) struct MatchArmWithNoBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " => todo!(),", applicability = "has-placeholders")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_never_pattern_with_body)]
pub(crate) struct NeverPatternWithBody {
    #[primary_span]
    #[label]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_never_pattern_with_guard)]
pub(crate) struct NeverPatternWithGuard {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_arbitrary_expression_in_pattern)]
pub(crate) struct ArbitraryExpressionInPattern {
    #[primary_span]
    pub span: Span,
    #[note(ast_lowering_pattern_from_macro_note)]
    pub pattern_from_macro_note: bool,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_inclusive_range_with_no_end)]
pub(crate) struct InclusiveRangeWithNoEnd {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    ast_lowering_bad_return_type_notation_output_suggestion,
    applicability = "machine-applicable",
    style = "verbose"
)]
/// Given `T: Tr<m() -> Ret>` or `T: Tr<m(Ty) -> Ret>`, suggest `T: Tr<m(..)>`.
pub(crate) struct RTNSuggestion {
    #[suggestion_part(code = "")]
    pub output: Span,
    #[suggestion_part(code = "(..)")]
    pub input: Span,
}

#[derive(Diagnostic)]
pub(crate) enum BadReturnTypeNotation {
    #[diag(ast_lowering_bad_return_type_notation_inputs)]
    Inputs {
        #[primary_span]
        #[suggestion(code = "(..)", applicability = "machine-applicable", style = "verbose")]
        span: Span,
    },
    #[diag(ast_lowering_bad_return_type_notation_output)]
    Output {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        suggestion: RTNSuggestion,
    },
    #[diag(ast_lowering_bad_return_type_notation_needs_dots)]
    NeedsDots {
        #[primary_span]
        #[suggestion(code = "(..)", applicability = "machine-applicable", style = "verbose")]
        span: Span,
    },
    #[diag(ast_lowering_bad_return_type_notation_position)]
    Position {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(ast_lowering_generic_param_default_in_binder)]
pub(crate) struct GenericParamDefaultInBinder {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_async_bound_not_on_trait)]
pub(crate) struct AsyncBoundNotOnTrait {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_async_bound_only_for_fn_traits)]
pub(crate) struct AsyncBoundOnlyForFnTraits {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_no_precise_captures_on_apit)]
pub(crate) struct NoPreciseCapturesOnApit {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_yield_in_closure)]
pub(crate) struct YieldInClosure {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "#[coroutine] ", applicability = "maybe-incorrect", style = "verbose")]
    pub suggestion: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_legacy_const_generic_arg)]
pub(crate) struct InvalidLegacyConstGenericArg {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: UseConstGenericArg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    ast_lowering_invalid_legacy_const_generic_arg_suggestion,
    applicability = "maybe-incorrect"
)]
pub(crate) struct UseConstGenericArg {
    #[suggestion_part(code = "::<{const_args}>")]
    pub end_of_fn: Span,
    pub const_args: String,
    pub other_args: String,
    #[suggestion_part(code = "{other_args}")]
    pub call_args: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_union_default_field_values)]
pub(crate) struct UnionWithDefault {
    #[primary_span]
    pub span: Span,
}
