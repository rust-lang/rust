use rustc_macros::SessionDiagnostic;
use rustc_span::symbol::MacroRulesNormalizedIdent;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(expand::expr_repeat_no_syntax_vars)]
pub(crate) struct NoSyntaxVarsExprRepeat {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(expand::must_repeat_once)]
pub(crate) struct MustRepeatOnce {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(expand::count_repetition_misplaced)]
pub(crate) struct CountRepetitionMisplaced {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(expand::meta_var_expr_unrecognized_var)]
pub(crate) struct MetaVarExprUnrecognizedVar {
    #[primary_span]
    pub span: Span,
    pub key: MacroRulesNormalizedIdent,
}

#[derive(SessionDiagnostic)]
#[diag(expand::var_still_repeating)]
pub(crate) struct VarStillRepeating {
    #[primary_span]
    pub span: Span,
    pub ident: MacroRulesNormalizedIdent,
}

#[derive(SessionDiagnostic)]
#[diag(expand::meta_var_dif_seq_matchers)]
pub(crate) struct MetaVarsDifSeqMatchers {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}
