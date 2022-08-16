use rustc_errors::{fluent, AddSubdiagnostic, Applicability, Diagnostic};
use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

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
