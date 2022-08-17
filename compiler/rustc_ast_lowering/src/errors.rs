use rustc_errors::{fluent, AddSubdiagnostic, Applicability, Diagnostic};
use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic, Clone, Copy)]
#[error(ast_lowering::generic_type_with_parentheses, code = "E0214")]
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
#[error(ast_lowering::invalid_abi, code = "E0703")]
pub struct InvalidAbi {
    #[primary_span]
    #[label]
    pub span: Span,
    pub abi: Symbol,
    pub valid_abis: String,
}

#[derive(SessionDiagnostic, Clone, Copy)]
#[error(ast_lowering::assoc_ty_parentheses)]
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
#[error(ast_lowering::misplaced_impl_trait, code = "E0562")]
pub struct MisplacedImplTrait {
    #[primary_span]
    pub span: Span,
    pub position: String,
}
