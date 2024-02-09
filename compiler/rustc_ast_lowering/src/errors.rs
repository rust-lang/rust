use rustc_errors::{
    codes::*, AddToDiagnostic, Diagnostic, DiagnosticArgFromDisplay, SubdiagnosticMessageOp,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_generic_type_with_parentheses, code = E0214)]
pub struct GenericTypeWithParentheses {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<UseAngleBrackets>,
}

#[derive(Clone, Copy, Subdiagnostic)]
#[multipart_suggestion(ast_lowering_use_angle_brackets, applicability = "maybe-incorrect")]
pub struct UseAngleBrackets {
    #[suggestion_part(code = "<")]
    pub open_param: Span,
    #[suggestion_part(code = ">")]
    pub close_param: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_abi, code = E0703)]
#[note]
pub struct InvalidAbi {
    #[primary_span]
    #[label]
    pub span: Span,
    pub abi: Symbol,
    pub command: String,
    #[subdiagnostic]
    pub explain: Option<InvalidAbiReason>,
    #[subdiagnostic]
    pub suggestion: Option<InvalidAbiSuggestion>,
}

pub struct InvalidAbiReason(pub &'static str);

impl AddToDiagnostic for InvalidAbiReason {
    fn add_to_diagnostic_with<F: SubdiagnosticMessageOp>(self, diag: &mut Diagnostic, _: F) {
        #[allow(rustc::untranslatable_diagnostic)]
        diag.note(self.0);
    }
}

#[derive(Subdiagnostic)]
#[suggestion(
    ast_lowering_invalid_abi_suggestion,
    code = "{suggestion}",
    applicability = "maybe-incorrect"
)]
pub struct InvalidAbiSuggestion {
    #[primary_span]
    pub span: Span,
    pub suggestion: String,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_assoc_ty_parentheses)]
pub struct AssocTyParentheses {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: AssocTyParenthesesSub,
}

#[derive(Clone, Copy, Subdiagnostic)]
pub enum AssocTyParenthesesSub {
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
pub struct MisplacedImplTrait<'a> {
    #[primary_span]
    pub span: Span,
    pub position: DiagnosticArgFromDisplay<'a>,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_misplaced_assoc_ty_binding)]
pub struct MisplacedAssocTyBinding<'a> {
    #[primary_span]
    pub span: Span,
    pub position: DiagnosticArgFromDisplay<'a>,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_underscore_expr_lhs_assign)]
pub struct UnderscoreExprLhsAssign {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_base_expression_double_dot, code = E0797)]
pub struct BaseExpressionDoubleDot {
    #[primary_span]
    #[suggestion(code = "/* expr */", applicability = "has-placeholders", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_await_only_in_async_fn_and_blocks, code = E0728)]
pub struct AwaitOnlyInAsyncFnAndBlocks {
    #[primary_span]
    #[label]
    pub await_kw_span: Span,
    #[label(ast_lowering_this_not_async)]
    pub item_span: Option<Span>,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_coroutine_too_many_parameters, code = E0628)]
pub struct CoroutineTooManyParameters {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_closure_cannot_be_static, code = E0697)]
pub struct ClosureCannotBeStatic {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_functional_record_update_destructuring_assignment)]
pub struct FunctionalRecordUpdateDestructuringAssignment {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_async_coroutines_not_supported, code = E0727)]
pub struct AsyncCoroutinesNotSupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_inline_asm_unsupported_target, code = E0472)]
pub struct InlineAsmUnsupportedTarget {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_att_syntax_only_x86)]
pub struct AttSyntaxOnlyX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_abi_specified_multiple_times)]
pub struct AbiSpecifiedMultipleTimes {
    #[primary_span]
    pub abi_span: Span,
    pub prev_name: Symbol,
    #[label]
    pub prev_span: Span,
    #[note]
    pub equivalent: Option<()>,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_clobber_abi_not_supported)]
pub struct ClobberAbiNotSupported {
    #[primary_span]
    pub abi_span: Span,
}

#[derive(Diagnostic)]
#[note]
#[diag(ast_lowering_invalid_abi_clobber_abi)]
pub struct InvalidAbiClobberAbi {
    #[primary_span]
    pub abi_span: Span,
    pub supported_abis: String,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_invalid_register)]
pub struct InvalidRegister<'a> {
    #[primary_span]
    pub op_span: Span,
    pub reg: Symbol,
    pub error: &'a str,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_invalid_register_class)]
pub struct InvalidRegisterClass<'a> {
    #[primary_span]
    pub op_span: Span,
    pub reg_class: Symbol,
    pub error: &'a str,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_invalid_asm_template_modifier_reg_class)]
pub struct InvalidAsmTemplateModifierRegClass {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
    #[subdiagnostic]
    pub sub: InvalidAsmTemplateModifierRegClassSub,
}

#[derive(Subdiagnostic)]
pub enum InvalidAsmTemplateModifierRegClassSub {
    #[note(ast_lowering_support_modifiers)]
    SupportModifier { class_name: Symbol, modifiers: String },
    #[note(ast_lowering_does_not_support_modifiers)]
    DoesNotSupportModifier { class_name: Symbol },
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_invalid_asm_template_modifier_const)]
pub struct InvalidAsmTemplateModifierConst {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_invalid_asm_template_modifier_sym)]
pub struct InvalidAsmTemplateModifierSym {
    #[primary_span]
    #[label(ast_lowering_template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering_argument)]
    pub op_span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_register_class_only_clobber)]
pub struct RegisterClassOnlyClobber {
    #[primary_span]
    pub op_span: Span,
    pub reg_class_name: Symbol,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_register_conflict)]
pub struct RegisterConflict<'a> {
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

#[derive(Diagnostic, Clone, Copy)]
#[help]
#[diag(ast_lowering_sub_tuple_binding)]
pub struct SubTupleBinding<'a> {
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

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_extra_double_dot)]
pub struct ExtraDoubleDot<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(ast_lowering_previously_used_here)]
    pub prev_span: Span,
    pub ctx: &'a str,
}

#[derive(Diagnostic, Clone, Copy)]
#[note]
#[diag(ast_lowering_misplaced_double_dot)]
pub struct MisplacedDoubleDot {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_misplaced_relax_trait_bound)]
pub struct MisplacedRelaxTraitBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_match_arm_with_no_body)]
pub struct MatchArmWithNoBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " => todo!(),", applicability = "has-placeholders")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_never_pattern_with_body)]
pub struct NeverPatternWithBody {
    #[primary_span]
    #[label]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(ast_lowering_never_pattern_with_guard)]
pub struct NeverPatternWithGuard {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_arbitrary_expression_in_pattern)]
pub struct ArbitraryExpressionInPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic, Clone, Copy)]
#[diag(ast_lowering_inclusive_range_with_no_end)]
pub struct InclusiveRangeWithNoEnd {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub enum BadReturnTypeNotation {
    #[diag(ast_lowering_bad_return_type_notation_inputs)]
    Inputs {
        #[primary_span]
        #[suggestion(code = "()", applicability = "maybe-incorrect")]
        span: Span,
    },
    #[diag(ast_lowering_bad_return_type_notation_output)]
    Output {
        #[primary_span]
        #[suggestion(code = "", applicability = "maybe-incorrect")]
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
