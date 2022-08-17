use rustc_errors::{fluent, AddSubdiagnostic, Applicability, Diagnostic};
use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

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
pub struct MisplacedImplTrait {
    #[primary_span]
    pub span: Span,
    pub position: String,
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
