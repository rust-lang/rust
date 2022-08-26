use rustc_errors::{fluent, AddSubdiagnostic, Applicability, Diagnostic, DiagnosticArgFromDisplay};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::generic_type_with_parentheses, code = "E0214")]
pub struct GenericTypeWithParentheses {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<UseAngleBrackets>,
}

#[derive(Clone, Copy)]
pub struct UseAngleBrackets {
    pub open_param: Span,
    pub close_param: Span,
}

impl AddSubdiagnostic for UseAngleBrackets {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        diag.multipart_suggestion(
            fluent::ast_lowering::use_angle_brackets,
            vec![(self.open_param, String::from("<")), (self.close_param, String::from(">"))],
            Applicability::MaybeIncorrect,
        );
    }
}

#[derive(SessionDiagnostic)]
#[help]
#[diag(ast_lowering::invalid_abi, code = "E0703")]
pub struct InvalidAbi {
    #[primary_span]
    #[label]
    pub span: Span,
    pub abi: Symbol,
    pub valid_abis: String,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::assoc_ty_parentheses)]
pub struct AssocTyParentheses {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: AssocTyParenthesesSub,
}

#[derive(Clone, Copy)]
pub enum AssocTyParenthesesSub {
    Empty { parentheses_span: Span },
    NotEmpty { open_param: Span, close_param: Span },
}

impl AddSubdiagnostic for AssocTyParenthesesSub {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        match self {
            Self::Empty { parentheses_span } => diag.multipart_suggestion(
                fluent::ast_lowering::remove_parentheses,
                vec![(parentheses_span, String::new())],
                Applicability::MaybeIncorrect,
            ),
            Self::NotEmpty { open_param, close_param } => diag.multipart_suggestion(
                fluent::ast_lowering::use_angle_brackets,
                vec![(open_param, String::from("<")), (close_param, String::from(">"))],
                Applicability::MaybeIncorrect,
            ),
        };
    }
}

#[derive(SessionDiagnostic)]
#[diag(ast_lowering::misplaced_impl_trait, code = "E0562")]
pub struct MisplacedImplTrait<'a> {
    #[primary_span]
    pub span: Span,
    pub position: DiagnosticArgFromDisplay<'a>,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::rustc_box_attribute_error)]
pub struct RustcBoxAttributeError {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::underscore_expr_lhs_assign)]
pub struct UnderscoreExprLhsAssign {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::base_expression_double_dot)]
pub struct BaseExpressionDoubleDot {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::await_only_in_async_fn_and_blocks, code = "E0728")]
pub struct AwaitOnlyInAsyncFnAndBlocks {
    #[primary_span]
    #[label]
    pub dot_await_span: Span,
    #[label(ast_lowering::this_not_async)]
    pub item_span: Option<Span>,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::generator_too_many_parameters, code = "E0628")]
pub struct GeneratorTooManyParameters {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::closure_cannot_be_static, code = "E0697")]
pub struct ClosureCannotBeStatic {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[help]
#[diag(ast_lowering::async_non_move_closure_not_supported, code = "E0708")]
pub struct AsyncNonMoveClosureNotSupported {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::functional_record_update_destructuring_assignment)]
pub struct FunctionalRecordUpdateDestructuringAssignemnt {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::async_generators_not_supported, code = "E0727")]
pub struct AsyncGeneratorsNotSupported {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::inline_asm_unsupported_target, code = "E0472")]
pub struct InlineAsmUnsupportedTarget {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::att_syntax_only_x86)]
pub struct AttSyntaxOnlyX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::abi_specified_multiple_times)]
pub struct AbiSpecifiedMultipleTimes {
    #[primary_span]
    pub abi_span: Span,
    pub prev_name: Symbol,
    #[label]
    pub prev_span: Span,
    #[note]
    pub equivalent: Option<()>,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::clobber_abi_not_supported)]
pub struct ClobberAbiNotSupported {
    #[primary_span]
    pub abi_span: Span,
}

#[derive(SessionDiagnostic)]
#[note]
#[diag(ast_lowering::invalid_abi_clobber_abi)]
pub struct InvalidAbiClobberAbi {
    #[primary_span]
    pub abi_span: Span,
    pub supported_abis: String,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::invalid_register)]
pub struct InvalidRegister<'a> {
    #[primary_span]
    pub op_span: Span,
    pub reg: Symbol,
    pub error: &'a str,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::invalid_register_class)]
pub struct InvalidRegisterClass<'a> {
    #[primary_span]
    pub op_span: Span,
    pub reg_class: Symbol,
    pub error: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(ast_lowering::invalid_asm_template_modifier_reg_class)]
pub struct InvalidAsmTemplateModifierRegClass {
    #[primary_span]
    #[label(ast_lowering::template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering::argument)]
    pub op_span: Span,
    #[subdiagnostic]
    pub sub: InvalidAsmTemplateModifierRegClassSub,
}

#[derive(SessionSubdiagnostic)]
pub enum InvalidAsmTemplateModifierRegClassSub {
    #[note(ast_lowering::support_modifiers)]
    SupportModifier { class_name: Symbol, modifiers: String },
    #[note(ast_lowering::does_not_support_modifiers)]
    DoesNotSupportModifier { class_name: Symbol },
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::invalid_asm_template_modifier_const)]
pub struct InvalidAsmTemplateModifierConst {
    #[primary_span]
    #[label(ast_lowering::template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering::argument)]
    pub op_span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::invalid_asm_template_modifier_sym)]
pub struct InvalidAsmTemplateModifierSym {
    #[primary_span]
    #[label(ast_lowering::template_modifier)]
    pub placeholder_span: Span,
    #[label(ast_lowering::argument)]
    pub op_span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::register_class_only_clobber)]
pub struct RegisterClassOnlyClobber {
    #[primary_span]
    pub op_span: Span,
    pub reg_class_name: Symbol,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::register_conflict)]
pub struct RegisterConflict<'a> {
    #[primary_span]
    #[label(ast_lowering::register1)]
    pub op_span1: Span,
    #[label(ast_lowering::register2)]
    pub op_span2: Span,
    pub reg1_name: &'a str,
    pub reg2_name: &'a str,
    #[help]
    pub in_out: Option<Span>,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[help]
#[diag(ast_lowering::sub_tuple_binding)]
pub struct SubTupleBinding<'a> {
    #[primary_span]
    #[label]
    #[suggestion_verbose(
        ast_lowering::sub_tuple_binding_suggestion,
        code = "..",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
    pub ident: Ident,
    pub ident_name: Symbol,
    pub ctx: &'a str,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::extra_double_dot)]
pub struct ExtraDoubleDot<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(ast_lowering::previously_used_here)]
    pub prev_span: Span,
    pub ctx: &'a str,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[note]
#[diag(ast_lowering::misplaced_double_dot)]
pub struct MisplacedDoubleDot {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::misplaced_relax_trait_bound)]
pub struct MisplacedRelaxTraitBound {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::not_supported_for_lifetime_binder_async_closure)]
pub struct NotSupportedForLifetimeBinderAsyncClosure {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[diag(ast_lowering::arbitrary_expression_in_pattern)]
pub struct ArbitraryExpressionInPattern {
    #[primary_span]
    pub span: Span,
}
