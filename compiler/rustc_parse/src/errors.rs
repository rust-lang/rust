use std::borrow::Cow;

use rustc_ast::token::Token;
use rustc_ast::{Path, Visibility};
use rustc_errors::{
    codes::*, Applicability, Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level,
    SubdiagMessageOp, Subdiagnostic,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::edition::{Edition, LATEST_STABLE_EDITION};
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol};

use crate::fluent_generated as fluent;
use crate::parser::{ForbiddenLetReason, TokenDescription};

#[derive(Diagnostic)]
#[diag(parse_maybe_report_ambiguous_plus)]
pub(crate) struct AmbiguousPlus {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: AddParen,
}

#[derive(Diagnostic)]
#[diag(parse_maybe_recover_from_bad_type_plus, code = E0178)]
pub(crate) struct BadTypePlus {
    pub ty: String,
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: BadTypePlusSub,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_add_paren, applicability = "machine-applicable")]
pub(crate) struct AddParen {
    #[suggestion_part(code = "(")]
    pub lo: Span,
    #[suggestion_part(code = ")")]
    pub hi: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum BadTypePlusSub {
    AddParen {
        #[subdiagnostic]
        suggestion: AddParen,
    },
    #[label(parse_forgot_paren)]
    ForgotParen {
        #[primary_span]
        span: Span,
    },
    #[label(parse_expect_path)]
    ExpectPath {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parse_maybe_recover_from_bad_qpath_stage_2)]
pub(crate) struct BadQPathStage2 {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub wrap: WrapType,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct WrapType {
    #[suggestion_part(code = "<")]
    pub lo: Span,
    #[suggestion_part(code = ">")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(parse_incorrect_semicolon)]
pub(crate) struct IncorrectSemicolon<'a> {
    #[primary_span]
    #[suggestion(style = "verbose", code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[help]
    pub show_help: bool,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(parse_incorrect_use_of_await)]
pub(crate) struct IncorrectUseOfAwait {
    #[primary_span]
    #[suggestion(
        parse_parentheses_suggestion,
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_incorrect_use_of_await_postfix_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct AwaitSuggestion {
    #[suggestion_part(code = "")]
    pub removal: Span,
    #[suggestion_part(code = ".await{question_mark}")]
    pub dot_await: Span,
    pub question_mark: &'static str,
}

#[derive(Diagnostic)]
#[diag(parse_incorrect_use_of_await)]
pub(crate) struct IncorrectAwait {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: AwaitSuggestion,
}

#[derive(Diagnostic)]
#[diag(parse_in_in_typo)]
pub(crate) struct InInTypo {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "", style = "verbose", applicability = "machine-applicable")]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_variable_declaration)]
pub(crate) struct InvalidVariableDeclaration {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: InvalidVariableDeclarationSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidVariableDeclarationSub {
    #[suggestion(
        parse_switch_mut_let_order,
        style = "verbose",
        applicability = "maybe-incorrect",
        code = "let mut"
    )]
    SwitchMutLetOrder(#[primary_span] Span),
    #[suggestion(
        parse_missing_let_before_mut,
        applicability = "machine-applicable",
        style = "verbose",
        code = "let mut"
    )]
    MissingLet(#[primary_span] Span),
    #[suggestion(
        parse_use_let_not_auto,
        style = "verbose",
        applicability = "machine-applicable",
        code = "let"
    )]
    UseLetNotAuto(#[primary_span] Span),
    #[suggestion(
        parse_use_let_not_var,
        style = "verbose",
        applicability = "machine-applicable",
        code = "let"
    )]
    UseLetNotVar(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_switch_ref_box_order)]
pub(crate) struct SwitchRefBoxOrder {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", style = "verbose", code = "box ref")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_comparison_operator)]
pub(crate) struct InvalidComparisonOperator {
    #[primary_span]
    pub span: Span,
    pub invalid: String,
    #[subdiagnostic]
    pub sub: InvalidComparisonOperatorSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidComparisonOperatorSub {
    #[suggestion(
        parse_use_instead,
        style = "verbose",
        applicability = "machine-applicable",
        code = "{correct}"
    )]
    Correctable {
        #[primary_span]
        span: Span,
        invalid: String,
        correct: String,
    },
    #[label(parse_spaceship_operator_invalid)]
    Spaceship(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_invalid_logical_operator)]
#[note]
pub(crate) struct InvalidLogicalOperator {
    #[primary_span]
    pub span: Span,
    pub incorrect: String,
    #[subdiagnostic]
    pub sub: InvalidLogicalOperatorSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidLogicalOperatorSub {
    #[suggestion(
        parse_use_amp_amp_for_conjunction,
        style = "verbose",
        applicability = "machine-applicable",
        code = "&&"
    )]
    Conjunction(#[primary_span] Span),
    #[suggestion(
        parse_use_pipe_pipe_for_disjunction,
        style = "verbose",
        applicability = "machine-applicable",
        code = "||"
    )]
    Disjunction(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_tilde_is_not_unary_operator)]
pub(crate) struct TildeAsUnaryOperator(
    #[primary_span]
    #[suggestion(style = "verbose", applicability = "machine-applicable", code = "!")]
    pub Span,
);

#[derive(Diagnostic)]
#[diag(parse_unexpected_token_after_not)]
pub(crate) struct NotAsNegationOperator {
    #[primary_span]
    pub negated: Span,
    pub negated_desc: String,
    #[subdiagnostic]
    pub sub: NotAsNegationOperatorSub,
}

#[derive(Subdiagnostic)]
pub enum NotAsNegationOperatorSub {
    #[suggestion(
        parse_unexpected_token_after_not_default,
        style = "verbose",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotDefault(#[primary_span] Span),

    #[suggestion(
        parse_unexpected_token_after_not_bitwise,
        style = "verbose",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotBitwise(#[primary_span] Span),

    #[suggestion(
        parse_unexpected_token_after_not_logical,
        style = "verbose",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotLogical(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_malformed_loop_label)]
pub(crate) struct MalformedLoopLabel {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable", code = "'", style = "verbose")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(parse_lifetime_in_borrow_expression)]
pub(crate) struct LifetimeInBorrowExpression {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable", code = "", style = "verbose")]
    #[label]
    pub lifetime_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_field_expression_with_generic)]
pub(crate) struct FieldExpressionWithGeneric(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(parse_macro_invocation_with_qualified_path)]
pub(crate) struct MacroInvocationWithQualifiedPath(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(parse_unexpected_token_after_label)]
pub(crate) struct UnexpectedTokenAfterLabel {
    #[primary_span]
    #[label(parse_unexpected_token_after_label)]
    pub span: Span,
    #[suggestion(parse_suggestion_remove_label, style = "verbose", code = "")]
    pub remove_label: Option<Span>,
    #[subdiagnostic]
    pub enclose_in_block: Option<UnexpectedTokenAfterLabelSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion_enclose_in_block, applicability = "machine-applicable")]
pub(crate) struct UnexpectedTokenAfterLabelSugg {
    #[suggestion_part(code = "{{ ")]
    pub left: Span,
    #[suggestion_part(code = " }}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_require_colon_after_labeled_expression)]
#[note]
pub(crate) struct RequireColonAfterLabeledExpression {
    #[primary_span]
    pub span: Span,
    #[label]
    pub label: Span,
    #[suggestion(style = "verbose", applicability = "machine-applicable", code = ": ")]
    pub label_end: Span,
}

#[derive(Diagnostic)]
#[diag(parse_do_catch_syntax_removed)]
#[note]
pub(crate) struct DoCatchSyntaxRemoved {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "try", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_float_literal_requires_integer_part)]
pub(crate) struct FloatLiteralRequiresIntegerPart {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable", code = "0", style = "verbose")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(parse_missing_semicolon_before_array)]
pub(crate) struct MissingSemicolonBeforeArray {
    #[primary_span]
    pub open_delim: Span,
    #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = ";")]
    pub semicolon: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expect_dotdot_not_dotdotdot)]
pub(crate) struct MissingDotDot {
    #[primary_span]
    pub token_span: Span,
    #[suggestion(applicability = "maybe-incorrect", code = "..", style = "verbose")]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_block_macro_segment)]
pub(crate) struct InvalidBlockMacroSegment {
    #[primary_span]
    pub span: Span,
    #[label]
    pub context: Span,
    #[subdiagnostic]
    pub wrap: WrapInExplicitBlock,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct WrapInExplicitBlock {
    #[suggestion_part(code = "{{ ")]
    pub lo: Span,
    #[suggestion_part(code = " }}")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(parse_if_expression_missing_then_block)]
pub(crate) struct IfExpressionMissingThenBlock {
    #[primary_span]
    pub if_span: Span,
    #[subdiagnostic]
    pub missing_then_block_sub: IfExpressionMissingThenBlockSub,
    #[subdiagnostic]
    pub let_else_sub: Option<IfExpressionLetSomeSub>,
}

#[derive(Subdiagnostic)]
pub(crate) enum IfExpressionMissingThenBlockSub {
    #[help(parse_condition_possibly_unfinished)]
    UnfinishedCondition(#[primary_span] Span),
    #[help(parse_add_then_block)]
    AddThenBlock(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_ternary_operator)]
#[help]
pub struct TernaryOperator {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    parse_extra_if_in_let_else,
    applicability = "maybe-incorrect",
    code = "",
    style = "verbose"
)]
pub(crate) struct IfExpressionLetSomeSub {
    #[primary_span]
    pub if_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_if_expression_missing_condition)]
pub(crate) struct IfExpressionMissingCondition {
    #[primary_span]
    #[label(parse_condition_label)]
    pub if_span: Span,
    #[label(parse_block_label)]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_expression_found_let)]
#[note]
pub(crate) struct ExpectedExpressionFoundLet {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub reason: ForbiddenLetReason,
    #[subdiagnostic]
    pub missing_let: Option<MaybeMissingLet>,
    #[subdiagnostic]
    pub comparison: Option<MaybeComparison>,
}

#[derive(Subdiagnostic, Clone, Copy)]
#[multipart_suggestion(
    parse_maybe_missing_let,
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct MaybeMissingLet {
    #[suggestion_part(code = "let ")]
    pub span: Span,
}

#[derive(Subdiagnostic, Clone, Copy)]
#[multipart_suggestion(
    parse_maybe_comparison,
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct MaybeComparison {
    #[suggestion_part(code = "=")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expect_eq_instead_of_eqeq)]
pub(crate) struct ExpectedEqForLetExpr {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "maybe-incorrect", code = "=", style = "verbose")]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_else_block)]
pub(crate) struct ExpectedElseBlock {
    #[primary_span]
    pub first_tok_span: Span,
    pub first_tok: String,
    #[label]
    pub else_span: Span,
    #[suggestion(applicability = "maybe-incorrect", code = "if ", style = "verbose")]
    pub condition_start: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_struct_field)]
pub(crate) struct ExpectedStructField {
    #[primary_span]
    #[label]
    pub span: Span,
    pub token: Token,
    #[label(parse_ident_label)]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_outer_attribute_not_allowed_on_if_else)]
pub(crate) struct OuterAttributeNotAllowedOnIfElse {
    #[primary_span]
    pub last: Span,

    #[label(parse_branch_label)]
    pub branch_span: Span,

    #[label(parse_ctx_label)]
    pub ctx_span: Span,
    pub ctx: String,

    #[suggestion(applicability = "machine-applicable", code = "", style = "verbose")]
    pub attributes: Span,
}

#[derive(Diagnostic)]
#[diag(parse_missing_in_in_for_loop)]
pub(crate) struct MissingInInForLoop {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: MissingInInForLoopSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum MissingInInForLoopSub {
    // Has been misleading, at least in the past (closed Issue #48492), thus maybe-incorrect
    #[suggestion(
        parse_use_in_not_of,
        style = "verbose",
        applicability = "maybe-incorrect",
        code = "in"
    )]
    InNotOf(#[primary_span] Span),
    #[suggestion(
        parse_add_in,
        style = "verbose",
        applicability = "maybe-incorrect",
        code = " in "
    )]
    AddIn(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_missing_expression_in_for_loop)]
pub(crate) struct MissingExpressionInForLoop {
    #[primary_span]
    #[suggestion(
        code = "/* expression */ ",
        applicability = "has-placeholders",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_loop_else)]
#[note]
pub(crate) struct LoopElseNotSupported {
    #[primary_span]
    pub span: Span,
    pub loop_kind: &'static str,
    #[label(parse_loop_keyword)]
    pub loop_kw: Span,
}

#[derive(Diagnostic)]
#[diag(parse_missing_comma_after_match_arm)]
pub(crate) struct MissingCommaAfterMatchArm {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = ",", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_catch_after_try)]
#[help]
pub(crate) struct CatchAfterTry {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_comma_after_base_struct)]
#[note]
pub(crate) struct CommaAfterBaseStruct {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "verbose", applicability = "machine-applicable", code = "")]
    pub comma: Span,
}

#[derive(Diagnostic)]
#[diag(parse_eq_field_init)]
pub(crate) struct EqFieldInit {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable", code = ":", style = "verbose")]
    pub eq: Span,
}

#[derive(Diagnostic)]
#[diag(parse_dotdotdot)]
pub(crate) struct DotDotDot {
    #[primary_span]
    #[suggestion(
        parse_suggest_exclusive_range,
        applicability = "maybe-incorrect",
        code = "..",
        style = "verbose"
    )]
    #[suggestion(
        parse_suggest_inclusive_range,
        applicability = "maybe-incorrect",
        code = "..=",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_left_arrow_operator)]
pub(crate) struct LeftArrowOperator {
    #[primary_span]
    #[suggestion(applicability = "maybe-incorrect", code = "< -", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_remove_let)]
pub(crate) struct RemoveLet {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_use_eq_instead)]
pub(crate) struct UseEqInstead {
    #[primary_span]
    #[suggestion(style = "verbose", applicability = "machine-applicable", code = "=")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_use_empty_block_not_semi)]
pub(crate) struct UseEmptyBlockNotSemi {
    #[primary_span]
    #[suggestion(style = "hidden", applicability = "machine-applicable", code = "{{}}")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_comparison_interpreted_as_generic)]
pub(crate) struct ComparisonInterpretedAsGeneric {
    #[primary_span]
    #[label(parse_label_comparison)]
    pub comparison: Span,
    pub r#type: Path,
    #[label(parse_label_args)]
    pub args: Span,
    #[subdiagnostic]
    pub suggestion: ComparisonOrShiftInterpretedAsGenericSugg,
}

#[derive(Diagnostic)]
#[diag(parse_shift_interpreted_as_generic)]
pub(crate) struct ShiftInterpretedAsGeneric {
    #[primary_span]
    #[label(parse_label_comparison)]
    pub shift: Span,
    pub r#type: Path,
    #[label(parse_label_args)]
    pub args: Span,
    #[subdiagnostic]
    pub suggestion: ComparisonOrShiftInterpretedAsGenericSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct ComparisonOrShiftInterpretedAsGenericSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_found_expr_would_be_stmt)]
pub(crate) struct FoundExprWouldBeStmt {
    #[primary_span]
    #[label]
    pub span: Span,
    pub token: Token,
    #[subdiagnostic]
    pub suggestion: ExprParenthesesNeeded,
}

#[derive(Diagnostic)]
#[diag(parse_leading_plus_not_supported)]
pub(crate) struct LeadingPlusNotSupported {
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion(
        parse_suggestion_remove_plus,
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub remove_plus: Option<Span>,
    #[subdiagnostic]
    pub add_parentheses: Option<ExprParenthesesNeeded>,
}

#[derive(Diagnostic)]
#[diag(parse_parentheses_with_struct_fields)]
pub(crate) struct ParenthesesWithStructFields {
    #[primary_span]
    pub span: Span,
    pub r#type: Path,
    #[subdiagnostic]
    pub braces_for_struct: BracesForStructLiteral,
    #[subdiagnostic]
    pub no_fields_for_fn: NoFieldsForFnCall,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion_braces_for_struct, applicability = "maybe-incorrect")]
pub(crate) struct BracesForStructLiteral {
    #[suggestion_part(code = " {{ ")]
    pub first: Span,
    #[suggestion_part(code = " }}")]
    pub second: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion_no_fields_for_fn, applicability = "maybe-incorrect")]
pub(crate) struct NoFieldsForFnCall {
    #[suggestion_part(code = "")]
    pub fields: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_labeled_loop_in_break)]
pub(crate) struct LabeledLoopInBreak {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: WrapInParentheses,
}

#[derive(Subdiagnostic)]

pub(crate) enum WrapInParentheses {
    #[multipart_suggestion(
        parse_sugg_wrap_expression_in_parentheses,
        applicability = "machine-applicable"
    )]
    Expression {
        #[suggestion_part(code = "(")]
        left: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
    #[multipart_suggestion(
        parse_sugg_wrap_macro_in_parentheses,
        applicability = "machine-applicable"
    )]
    MacroArgs {
        #[suggestion_part(code = "(")]
        left: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parse_array_brackets_instead_of_braces)]
pub(crate) struct ArrayBracketsInsteadOfSpaces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: ArrayBracketsInsteadOfSpacesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "maybe-incorrect")]
pub(crate) struct ArrayBracketsInsteadOfSpacesSugg {
    #[suggestion_part(code = "[")]
    pub left: Span,
    #[suggestion_part(code = "]")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_match_arm_body_without_braces)]
pub(crate) struct MatchArmBodyWithoutBraces {
    #[primary_span]
    #[label(parse_label_statements)]
    pub statements: Span,
    #[label(parse_label_arrow)]
    pub arrow: Span,
    pub num_statements: usize,
    #[subdiagnostic]
    pub sub: MatchArmBodyWithoutBracesSugg,
}

#[derive(Diagnostic)]
#[diag(parse_inclusive_range_extra_equals)]
#[note]
pub(crate) struct InclusiveRangeExtraEquals {
    #[primary_span]
    #[suggestion(
        parse_suggestion_remove_eq,
        style = "verbose",
        code = "..=",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_inclusive_range_match_arrow)]
pub(crate) struct InclusiveRangeMatchArrow {
    #[primary_span]
    pub arrow: Span,
    #[label]
    pub span: Span,
    #[suggestion(style = "verbose", code = " ", applicability = "machine-applicable")]
    pub after_pat: Span,
}

#[derive(Diagnostic)]
#[diag(parse_inclusive_range_no_end, code = E0586)]
#[note]
pub(crate) struct InclusiveRangeNoEnd {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        parse_suggestion_open_range,
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum MatchArmBodyWithoutBracesSugg {
    #[multipart_suggestion(parse_suggestion_add_braces, applicability = "machine-applicable")]
    AddBraces {
        #[suggestion_part(code = "{{ ")]
        left: Span,
        #[suggestion_part(code = " }}")]
        right: Span,
    },
    #[suggestion(
        parse_suggestion_use_comma_not_semicolon,
        code = ",",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    UseComma {
        #[primary_span]
        semicolon: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parse_struct_literal_not_allowed_here)]
pub(crate) struct StructLiteralNotAllowedHere {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: StructLiteralNotAllowedHereSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct StructLiteralNotAllowedHereSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_literal_suffix_on_tuple_index)]
pub(crate) struct InvalidLiteralSuffixOnTupleIndex {
    #[primary_span]
    #[label]
    pub span: Span,
    pub suffix: Symbol,
    #[help(parse_tuple_exception_line_1)]
    #[help(parse_tuple_exception_line_2)]
    #[help(parse_tuple_exception_line_3)]
    pub exception: Option<()>,
}

#[derive(Diagnostic)]
#[diag(parse_non_string_abi_literal)]
pub(crate) struct NonStringAbiLiteral {
    #[primary_span]
    #[suggestion(code = "\"C\"", applicability = "maybe-incorrect", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_mismatched_closing_delimiter)]
pub(crate) struct MismatchedClosingDelimiter {
    #[primary_span]
    pub spans: Vec<Span>,
    pub delimiter: String,
    #[label(parse_label_unmatched)]
    pub unmatched: Span,
    #[label(parse_label_opening_candidate)]
    pub opening_candidate: Option<Span>,
    #[label(parse_label_unclosed)]
    pub unclosed: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_incorrect_visibility_restriction, code = E0704)]
#[help]
pub(crate) struct IncorrectVisibilityRestriction {
    #[primary_span]
    #[suggestion(code = "in {inner_str}", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    pub inner_str: String,
}

#[derive(Diagnostic)]
#[diag(parse_assignment_else_not_allowed)]
pub(crate) struct AssignmentElseNotAllowed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_statement_after_outer_attr)]
pub(crate) struct ExpectedStatementAfterOuterAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_doc_comment_does_not_document_anything, code = E0585)]
#[help]
pub(crate) struct DocCommentDoesNotDocumentAnything {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = ",", applicability = "machine-applicable", style = "verbose")]
    pub missing_comma: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_const_let_mutually_exclusive)]
pub(crate) struct ConstLetMutuallyExclusive {
    #[primary_span]
    #[suggestion(code = "const", applicability = "maybe-incorrect", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_expression_in_let_else)]
pub(crate) struct InvalidExpressionInLetElse {
    #[primary_span]
    pub span: Span,
    pub operator: &'static str,
    #[subdiagnostic]
    pub sugg: WrapInParentheses,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_curly_in_let_else)]
pub(crate) struct InvalidCurlyInLetElse {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WrapInParentheses,
}

#[derive(Diagnostic)]
#[diag(parse_compound_assignment_expression_in_let)]
#[help]
pub(crate) struct CompoundAssignmentExpressionInLet {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "verbose", code = "", applicability = "maybe-incorrect")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(parse_suffixed_literal_in_attribute)]
#[help]
pub(crate) struct SuffixedLiteralInAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_meta_item)]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub token: Token,
    #[subdiagnostic]
    pub quote_ident_sugg: Option<InvalidMetaItemQuoteIdentSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_quote_ident_sugg, applicability = "machine-applicable")]
pub(crate) struct InvalidMetaItemQuoteIdentSugg {
    #[suggestion_part(code = "\"")]
    pub before: Span,
    #[suggestion_part(code = "\"")]
    pub after: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    parse_sugg_escape_identifier,
    style = "verbose",
    applicability = "maybe-incorrect",
    code = "r#"
)]
pub(crate) struct SuggEscapeIdentifier {
    #[primary_span]
    pub span: Span,
    pub ident_name: String,
}

#[derive(Subdiagnostic)]
#[suggestion(
    parse_sugg_remove_comma,
    applicability = "machine-applicable",
    code = "",
    style = "verbose"
)]
pub(crate) struct SuggRemoveComma {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    parse_sugg_add_let_for_stmt,
    style = "verbose",
    applicability = "maybe-incorrect",
    code = "let "
)]
pub(crate) struct SuggAddMissingLetStmt {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedIdentifierFound {
    #[label(parse_expected_identifier_found_reserved_identifier)]
    ReservedIdentifier(#[primary_span] Span),
    #[label(parse_expected_identifier_found_keyword)]
    Keyword(#[primary_span] Span),
    #[label(parse_expected_identifier_found_reserved_keyword)]
    ReservedKeyword(#[primary_span] Span),
    #[label(parse_expected_identifier_found_doc_comment)]
    DocComment(#[primary_span] Span),
    #[label(parse_expected_identifier)]
    Other(#[primary_span] Span),
}

impl ExpectedIdentifierFound {
    pub fn new(token_descr: Option<TokenDescription>, span: Span) -> Self {
        (match token_descr {
            Some(TokenDescription::ReservedIdentifier) => {
                ExpectedIdentifierFound::ReservedIdentifier
            }
            Some(TokenDescription::Keyword) => ExpectedIdentifierFound::Keyword,
            Some(TokenDescription::ReservedKeyword) => ExpectedIdentifierFound::ReservedKeyword,
            Some(TokenDescription::DocComment) => ExpectedIdentifierFound::DocComment,
            None => ExpectedIdentifierFound::Other,
        })(span)
    }
}

pub(crate) struct ExpectedIdentifier {
    pub span: Span,
    pub token: Token,
    pub suggest_raw: Option<SuggEscapeIdentifier>,
    pub suggest_remove_comma: Option<SuggRemoveComma>,
    pub help_cannot_start_number: Option<HelpIdentifierStartsWithNumber>,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for ExpectedIdentifier {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let token_descr = TokenDescription::from_token(&self.token);

        let mut diag = Diag::new(
            dcx,
            level,
            match token_descr {
                Some(TokenDescription::ReservedIdentifier) => {
                    fluent::parse_expected_identifier_found_reserved_identifier_str
                }
                Some(TokenDescription::Keyword) => {
                    fluent::parse_expected_identifier_found_keyword_str
                }
                Some(TokenDescription::ReservedKeyword) => {
                    fluent::parse_expected_identifier_found_reserved_keyword_str
                }
                Some(TokenDescription::DocComment) => {
                    fluent::parse_expected_identifier_found_doc_comment_str
                }
                None => fluent::parse_expected_identifier_found_str,
            },
        );
        diag.span(self.span);
        diag.arg("token", self.token);

        if let Some(sugg) = self.suggest_raw {
            sugg.add_to_diag(&mut diag);
        }

        ExpectedIdentifierFound::new(token_descr, self.span).add_to_diag(&mut diag);

        if let Some(sugg) = self.suggest_remove_comma {
            sugg.add_to_diag(&mut diag);
        }

        if let Some(help) = self.help_cannot_start_number {
            help.add_to_diag(&mut diag);
        }

        diag
    }
}

#[derive(Subdiagnostic)]
#[help(parse_invalid_identifier_with_leading_number)]
pub(crate) struct HelpIdentifierStartsWithNumber {
    #[primary_span]
    pub num_span: Span,
}

pub(crate) struct ExpectedSemi {
    pub span: Span,
    pub token: Token,

    pub unexpected_token_label: Option<Span>,
    pub sugg: ExpectedSemiSugg,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for ExpectedSemi {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let token_descr = TokenDescription::from_token(&self.token);

        let mut diag = Diag::new(
            dcx,
            level,
            match token_descr {
                Some(TokenDescription::ReservedIdentifier) => {
                    fluent::parse_expected_semi_found_reserved_identifier_str
                }
                Some(TokenDescription::Keyword) => fluent::parse_expected_semi_found_keyword_str,
                Some(TokenDescription::ReservedKeyword) => {
                    fluent::parse_expected_semi_found_reserved_keyword_str
                }
                Some(TokenDescription::DocComment) => {
                    fluent::parse_expected_semi_found_doc_comment_str
                }
                None => fluent::parse_expected_semi_found_str,
            },
        );
        diag.span(self.span);
        diag.arg("token", self.token);

        if let Some(unexpected_token_label) = self.unexpected_token_label {
            diag.span_label(unexpected_token_label, fluent::parse_label_unexpected_token);
        }

        self.sugg.add_to_diag(&mut diag);

        diag
    }
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedSemiSugg {
    #[suggestion(
        parse_sugg_change_this_to_semi,
        code = ";",
        applicability = "machine-applicable",
        style = "short"
    )]
    ChangeToSemi(#[primary_span] Span),
    #[suggestion(
        parse_sugg_add_semi,
        code = ";",
        applicability = "machine-applicable",
        style = "short"
    )]
    AddSemi(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parse_struct_literal_body_without_path)]
pub(crate) struct StructLiteralBodyWithoutPath {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: StructLiteralBodyWithoutPathSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "has-placeholders")]
pub(crate) struct StructLiteralBodyWithoutPathSugg {
    #[suggestion_part(code = "{{ SomeStruct ")]
    pub before: Span,
    #[suggestion_part(code = " }}")]
    pub after: Span,
}

#[derive(Diagnostic)]
#[diag(parse_struct_literal_needing_parens)]
pub(crate) struct StructLiteralNeedingParens {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: StructLiteralNeedingParensSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct StructLiteralNeedingParensSugg {
    #[suggestion_part(code = "(")]
    pub before: Span,
    #[suggestion_part(code = ")")]
    pub after: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unmatched_angle_brackets)]
pub(crate) struct UnmatchedAngleBrackets {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    pub num_extra_brackets: usize,
}

#[derive(Diagnostic)]
#[diag(parse_generic_parameters_without_angle_brackets)]
pub(crate) struct GenericParamsWithoutAngleBrackets {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: GenericParamsWithoutAngleBracketsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct GenericParamsWithoutAngleBracketsSugg {
    #[suggestion_part(code = "<")]
    pub left: Span,
    #[suggestion_part(code = ">")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_comparison_operators_cannot_be_chained)]
pub(crate) struct ComparisonOperatorsCannotBeChained {
    #[primary_span]
    pub span: Vec<Span>,
    #[suggestion(
        parse_sugg_turbofish_syntax,
        style = "verbose",
        code = "::",
        applicability = "maybe-incorrect"
    )]
    pub suggest_turbofish: Option<Span>,
    #[help(parse_sugg_turbofish_syntax)]
    #[help(parse_sugg_parentheses_for_function_args)]
    pub help_turbofish: Option<()>,
    #[subdiagnostic]
    pub chaining_sugg: Option<ComparisonOperatorsCannotBeChainedSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ComparisonOperatorsCannotBeChainedSugg {
    #[suggestion(
        parse_sugg_split_comparison,
        style = "verbose",
        code = " && {middle_term}",
        applicability = "maybe-incorrect"
    )]
    SplitComparison {
        #[primary_span]
        span: Span,
        middle_term: String,
    },
    #[multipart_suggestion(parse_sugg_parenthesize, applicability = "maybe-incorrect")]
    Parenthesize {
        #[suggestion_part(code = "(")]
        left: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parse_question_mark_in_type)]
pub(crate) struct QuestionMarkInType {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: QuestionMarkInTypeSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct QuestionMarkInTypeSugg {
    #[suggestion_part(code = "Option<")]
    pub left: Span,
    #[suggestion_part(code = ">")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_parentheses_in_for_head)]
pub(crate) struct ParenthesesInForHead {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: ParenthesesInForHeadSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct ParenthesesInForHeadSugg {
    #[suggestion_part(code = " ")]
    pub left: Span,
    #[suggestion_part(code = " ")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_parentheses_in_match_arm_pattern)]
pub(crate) struct ParenthesesInMatchPat {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: ParenthesesInMatchPatSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct ParenthesesInMatchPatSugg {
    #[suggestion_part(code = "")]
    pub left: Span,
    #[suggestion_part(code = "")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_doc_comment_on_param_type)]
pub(crate) struct DocCommentOnParamType {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_attribute_on_param_type)]
pub(crate) struct AttributeOnParamType {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_pattern_method_param_without_body, code = E0642)]
pub(crate) struct PatternMethodParamWithoutBody {
    #[primary_span]
    #[suggestion(code = "_", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_self_param_not_first)]
pub(crate) struct SelfParamNotFirst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_const_generic_without_braces)]
pub(crate) struct ConstGenericWithoutBraces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ConstGenericWithoutBracesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct ConstGenericWithoutBracesSugg {
    #[suggestion_part(code = "{{ ")]
    pub left: Span,
    #[suggestion_part(code = " }}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_const_param_declaration)]
pub(crate) struct UnexpectedConstParamDeclaration {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<UnexpectedConstParamDeclarationSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnexpectedConstParamDeclarationSugg {
    #[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
    AddParam {
        #[suggestion_part(code = "<{snippet}>")]
        impl_generics: Span,
        #[suggestion_part(code = "{ident}")]
        incorrect_decl: Span,
        snippet: String,
        ident: String,
    },
    #[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
    AppendParam {
        #[suggestion_part(code = ", {snippet}")]
        impl_generics_end: Span,
        #[suggestion_part(code = "{ident}")]
        incorrect_decl: Span,
        snippet: String,
        ident: String,
    },
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_const_in_generic_param)]
pub(crate) struct UnexpectedConstInGenericParam {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "verbose", code = "", applicability = "maybe-incorrect")]
    pub to_remove: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_async_move_order_incorrect)]
pub(crate) struct AsyncMoveOrderIncorrect {
    #[primary_span]
    #[suggestion(style = "verbose", code = "async move", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_double_colon_in_bound)]
pub(crate) struct DoubleColonInBound {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = ": ", applicability = "machine-applicable", style = "verbose")]
    pub between: Span,
}

#[derive(Diagnostic)]
#[diag(parse_fn_ptr_with_generics)]
pub(crate) struct FnPtrWithGenerics {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<FnPtrWithGenericsSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_misplaced_return_type,
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct MisplacedReturnType {
    #[suggestion_part(code = " {snippet}")]
    pub fn_params_end: Span,
    pub snippet: String,
    #[suggestion_part(code = "")]
    pub ret_ty_span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "maybe-incorrect")]
pub(crate) struct FnPtrWithGenericsSugg {
    #[suggestion_part(code = "{snippet}")]
    pub left: Span,
    pub snippet: String,
    #[suggestion_part(code = "")]
    pub right: Span,
    pub arity: usize,
    pub for_param_list_exists: bool,
}

pub(crate) struct FnTraitMissingParen {
    pub span: Span,
}

impl Subdiagnostic for FnTraitMissingParen {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: &F,
    ) {
        diag.span_label(self.span, crate::fluent_generated::parse_fn_trait_missing_paren);
        diag.span_suggestion_short(
            self.span.shrink_to_hi(),
            crate::fluent_generated::parse_add_paren,
            "()",
            Applicability::MachineApplicable,
        );
    }
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_if_with_if)]
pub(crate) struct UnexpectedIfWithIf(
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = " ", style = "verbose")]
    pub Span,
);

#[derive(Diagnostic)]
#[diag(parse_maybe_fn_typo_with_impl)]
pub(crate) struct FnTypoWithImpl {
    #[primary_span]
    #[suggestion(applicability = "maybe-incorrect", code = "impl", style = "verbose")]
    pub fn_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_fn_path_found_fn_keyword)]
pub(crate) struct ExpectedFnPathFoundFnKeyword {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "Fn", style = "verbose")]
    pub fn_token_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_path_single_colon)]
pub(crate) struct PathSingleColon {
    #[primary_span]
    pub span: Span,

    #[suggestion(applicability = "machine-applicable", code = ":", style = "verbose")]
    pub suggestion: Span,

    #[note(parse_type_ascription_removed)]
    pub type_ascription: Option<()>,
}

#[derive(Diagnostic)]
#[diag(parse_colon_as_semi)]
pub(crate) struct ColonAsSemi {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = ";", style = "verbose")]
    pub span: Span,

    #[note(parse_type_ascription_removed)]
    pub type_ascription: Option<()>,
}

#[derive(Diagnostic)]
#[diag(parse_where_clause_before_tuple_struct_body)]
pub(crate) struct WhereClauseBeforeTupleStructBody {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(parse_name_label)]
    pub name: Span,
    #[label(parse_body_label)]
    pub body: Span,
    #[subdiagnostic]
    pub sugg: Option<WhereClauseBeforeTupleStructBodySugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct WhereClauseBeforeTupleStructBodySugg {
    #[suggestion_part(code = "{snippet}")]
    pub left: Span,
    pub snippet: String,
    #[suggestion_part(code = "")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_async_fn_in_2015, code = E0670)]
pub(crate) struct AsyncFnIn2015 {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub help: HelpUseLatestEdition,
}

#[derive(Subdiagnostic)]
#[label(parse_async_block_in_2015)]
pub(crate) struct AsyncBlockIn2015 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_async_move_block_in_2015)]
pub(crate) struct AsyncMoveBlockIn2015 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_async_bound_modifier_in_2015)]
pub(crate) struct AsyncBoundModifierIn2015 {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub help: HelpUseLatestEdition,
}

#[derive(Diagnostic)]
#[diag(parse_self_argument_pointer)]
pub(crate) struct SelfArgumentPointer {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_token_after_dot)]
pub struct UnexpectedTokenAfterDot<'a> {
    #[primary_span]
    pub span: Span,
    pub actual: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag(parse_visibility_not_followed_by_item)]
#[help]
pub(crate) struct VisibilityNotFollowedByItem {
    #[primary_span]
    #[label]
    pub span: Span,
    pub vis: Visibility,
}

#[derive(Diagnostic)]
#[diag(parse_default_not_followed_by_item)]
#[note]
pub(crate) struct DefaultNotFollowedByItem {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum MissingKeywordForItemDefinition {
    #[diag(parse_missing_enum_for_enum_definition)]
    Enum {
        #[primary_span]
        span: Span,
        #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = "enum ")]
        insert_span: Span,
        ident: Ident,
    },
    #[diag(parse_missing_enum_or_struct_for_item_definition)]
    EnumOrStruct {
        #[primary_span]
        span: Span,
    },
    #[diag(parse_missing_struct_for_struct_definition)]
    Struct {
        #[primary_span]
        span: Span,
        #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = "struct ")]
        insert_span: Span,
        ident: Ident,
    },
    #[diag(parse_missing_fn_for_function_definition)]
    Function {
        #[primary_span]
        span: Span,
        #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = "fn ")]
        insert_span: Span,
        ident: Ident,
    },
    #[diag(parse_missing_fn_for_method_definition)]
    Method {
        #[primary_span]
        span: Span,
        #[suggestion(style = "verbose", applicability = "maybe-incorrect", code = "fn ")]
        insert_span: Span,
        ident: Ident,
    },
    #[diag(parse_missing_fn_or_struct_for_item_definition)]
    Ambiguous {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        subdiag: Option<AmbiguousMissingKwForItemSub>,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum AmbiguousMissingKwForItemSub {
    #[suggestion(
        parse_suggestion,
        applicability = "maybe-incorrect",
        code = "{snippet}!",
        style = "verbose"
    )]
    SuggestMacro {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[help(parse_help)]
    HelpMacro,
}

#[derive(Diagnostic)]
#[diag(parse_missing_fn_params)]
pub(crate) struct MissingFnParams {
    #[primary_span]
    #[suggestion(code = "()", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_missing_trait_in_trait_impl)]
pub(crate) struct MissingTraitInTraitImpl {
    #[primary_span]
    #[suggestion(
        parse_suggestion_add_trait,
        code = " Trait ",
        applicability = "has-placeholders",
        style = "verbose"
    )]
    pub span: Span,
    #[suggestion(
        parse_suggestion_remove_for,
        code = "",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub for_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_missing_for_in_trait_impl)]
pub(crate) struct MissingForInTraitImpl {
    #[primary_span]
    #[suggestion(style = "verbose", code = " for ", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_trait_in_trait_impl_found_type)]
pub(crate) struct ExpectedTraitInTraitImplFoundType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_extra_impl_keyword_in_trait_impl)]
pub(crate) struct ExtraImplKeywordInTraitImpl {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect", style = "short")]
    pub extra_impl_kw: Span,
    #[note]
    pub impl_trait_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_bounds_not_allowed_on_trait_aliases)]
pub(crate) struct BoundsNotAllowedOnTraitAliases {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_trait_alias_cannot_be_auto)]
pub(crate) struct TraitAliasCannotBeAuto {
    #[primary_span]
    #[label(parse_trait_alias_cannot_be_auto)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_trait_alias_cannot_be_unsafe)]
pub(crate) struct TraitAliasCannotBeUnsafe {
    #[primary_span]
    #[label(parse_trait_alias_cannot_be_unsafe)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_associated_static_item_not_allowed)]
pub(crate) struct AssociatedStaticItemNotAllowed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_extern_crate_name_with_dashes)]
pub(crate) struct ExternCrateNameWithDashes {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ExternCrateNameWithDashesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct ExternCrateNameWithDashesSugg {
    #[suggestion_part(code = "_")]
    pub dashes: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_extern_item_cannot_be_const)]
#[note]
pub(crate) struct ExternItemCannotBeConst {
    #[primary_span]
    pub ident_span: Span,
    #[suggestion(code = "static ", applicability = "machine-applicable", style = "verbose")]
    pub const_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_const_global_cannot_be_mutable)]
pub(crate) struct ConstGlobalCannotBeMutable {
    #[primary_span]
    #[label]
    pub ident_span: Span,
    #[suggestion(code = "static", style = "verbose", applicability = "maybe-incorrect")]
    pub const_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_missing_const_type)]
pub(crate) struct MissingConstType {
    #[primary_span]
    #[suggestion(code = "{colon} <type>", style = "verbose", applicability = "has-placeholders")]
    pub span: Span,

    pub kind: &'static str,
    pub colon: &'static str,
}

#[derive(Diagnostic)]
#[diag(parse_enum_struct_mutually_exclusive)]
pub(crate) struct EnumStructMutuallyExclusive {
    #[primary_span]
    #[suggestion(code = "enum", style = "verbose", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum UnexpectedTokenAfterStructName {
    #[diag(parse_unexpected_token_after_struct_name_found_reserved_identifier)]
    ReservedIdentifier {
        #[primary_span]
        #[label(parse_unexpected_token_after_struct_name)]
        span: Span,
        token: Token,
    },
    #[diag(parse_unexpected_token_after_struct_name_found_keyword)]
    Keyword {
        #[primary_span]
        #[label(parse_unexpected_token_after_struct_name)]
        span: Span,
        token: Token,
    },
    #[diag(parse_unexpected_token_after_struct_name_found_reserved_keyword)]
    ReservedKeyword {
        #[primary_span]
        #[label(parse_unexpected_token_after_struct_name)]
        span: Span,
        token: Token,
    },
    #[diag(parse_unexpected_token_after_struct_name_found_doc_comment)]
    DocComment {
        #[primary_span]
        #[label(parse_unexpected_token_after_struct_name)]
        span: Span,
        token: Token,
    },
    #[diag(parse_unexpected_token_after_struct_name_found_other)]
    Other {
        #[primary_span]
        #[label(parse_unexpected_token_after_struct_name)]
        span: Span,
        token: Token,
    },
}

impl UnexpectedTokenAfterStructName {
    pub fn new(span: Span, token: Token) -> Self {
        match TokenDescription::from_token(&token) {
            Some(TokenDescription::ReservedIdentifier) => Self::ReservedIdentifier { span, token },
            Some(TokenDescription::Keyword) => Self::Keyword { span, token },
            Some(TokenDescription::ReservedKeyword) => Self::ReservedKeyword { span, token },
            Some(TokenDescription::DocComment) => Self::DocComment { span, token },
            None => Self::Other { span, token },
        }
    }
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_self_in_generic_parameters)]
#[note]
pub(crate) struct UnexpectedSelfInGenericParameters {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_default_value_for_lifetime_in_generic_parameters)]
pub(crate) struct UnexpectedDefaultValueForLifetimeInGenericParameters {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_multiple_where_clauses)]
pub(crate) struct MultipleWhereClauses {
    #[primary_span]
    pub span: Span,
    #[label]
    pub previous: Span,
    #[suggestion(style = "verbose", code = ",", applicability = "maybe-incorrect")]
    pub between: Span,
}

#[derive(Diagnostic)]
pub(crate) enum UnexpectedNonterminal {
    #[diag(parse_nonterminal_expected_item_keyword)]
    Item(#[primary_span] Span),
    #[diag(parse_nonterminal_expected_statement)]
    Statement(#[primary_span] Span),
    #[diag(parse_nonterminal_expected_ident)]
    Ident {
        #[primary_span]
        span: Span,
        token: Token,
    },
    #[diag(parse_nonterminal_expected_lifetime)]
    Lifetime {
        #[primary_span]
        span: Span,
        token: Token,
    },
}

#[derive(Diagnostic)]
pub(crate) enum TopLevelOrPatternNotAllowed {
    #[diag(parse_or_pattern_not_allowed_in_let_binding)]
    LetBinding {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        sub: Option<TopLevelOrPatternNotAllowedSugg>,
    },
    #[diag(parse_or_pattern_not_allowed_in_fn_parameters)]
    FunctionParameter {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        sub: Option<TopLevelOrPatternNotAllowedSugg>,
    },
}

#[derive(Diagnostic)]
#[diag(parse_cannot_be_raw_ident)]
pub struct CannotBeRawIdent {
    #[primary_span]
    pub span: Span,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag(parse_keyword_lifetime)]
pub struct KeywordLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_label)]
pub struct InvalidLabel {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(parse_cr_doc_comment)]
pub struct CrDocComment {
    #[primary_span]
    pub span: Span,
    pub block: bool,
}

#[derive(Diagnostic)]
#[diag(parse_no_digits_literal, code = E0768)]
pub struct NoDigitsLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_digit_literal)]
pub struct InvalidDigitLiteral {
    #[primary_span]
    pub span: Span,
    pub base: u32,
}

#[derive(Diagnostic)]
#[diag(parse_empty_exponent_float)]
pub struct EmptyExponentFloat {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_float_literal_unsupported_base)]
pub struct FloatLiteralUnsupportedBase {
    #[primary_span]
    pub span: Span,
    pub base: &'static str,
}

#[derive(Diagnostic)]
#[diag(parse_unknown_prefix)]
#[note]
pub struct UnknownPrefix<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub prefix: &'a str,
    #[subdiagnostic]
    pub sugg: Option<UnknownPrefixSugg>,
}

#[derive(Subdiagnostic)]
#[note(parse_macro_expands_to_adt_field)]
pub struct MacroExpandsToAdtField<'a> {
    pub adt_ty: &'a str,
}

#[derive(Subdiagnostic)]
pub enum UnknownPrefixSugg {
    #[suggestion(
        parse_suggestion_br,
        code = "br",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    UseBr(#[primary_span] Span),
    #[suggestion(
        parse_suggestion_whitespace,
        code = " ",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Whitespace(#[primary_span] Span),
    #[multipart_suggestion(
        parse_suggestion_str,
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    MeantStr {
        #[suggestion_part(code = "\"")]
        start: Span,
        #[suggestion_part(code = "\"")]
        end: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parse_too_many_hashes)]
pub struct TooManyHashes {
    #[primary_span]
    pub span: Span,
    pub num: u32,
}

#[derive(Diagnostic)]
#[diag(parse_unknown_start_of_token)]
pub struct UnknownTokenStart {
    #[primary_span]
    pub span: Span,
    pub escaped: String,
    #[subdiagnostic]
    pub sugg: Option<TokenSubstitution>,
    #[subdiagnostic]
    pub null: Option<UnknownTokenNull>,
    #[subdiagnostic]
    pub repeat: Option<UnknownTokenRepeat>,
}

#[derive(Subdiagnostic)]
pub enum TokenSubstitution {
    #[suggestion(
        parse_sugg_quotes,
        code = "{suggestion}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    DirectedQuotes {
        #[primary_span]
        span: Span,
        suggestion: String,
        ascii_str: &'static str,
        ascii_name: &'static str,
    },
    #[suggestion(
        parse_sugg_other,
        code = "{suggestion}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Other {
        #[primary_span]
        span: Span,
        suggestion: String,
        ch: String,
        u_name: &'static str,
        ascii_str: &'static str,
        ascii_name: &'static str,
    },
}

#[derive(Subdiagnostic)]
#[note(parse_note_repeats)]
pub struct UnknownTokenRepeat {
    pub repeats: usize,
}

#[derive(Subdiagnostic)]
#[help(parse_help_null)]
pub struct UnknownTokenNull;

#[derive(Diagnostic)]
pub enum UnescapeError {
    #[diag(parse_invalid_unicode_escape)]
    #[help]
    InvalidUnicodeEscape {
        #[primary_span]
        #[label]
        span: Span,
        surrogate: bool,
    },
    #[diag(parse_escape_only_char)]
    EscapeOnlyChar {
        #[primary_span]
        span: Span,
        #[suggestion(
            parse_escape,
            applicability = "machine-applicable",
            code = "{escaped_sugg}",
            style = "verbose"
        )]
        char_span: Span,
        escaped_sugg: String,
        escaped_msg: String,
        byte: bool,
    },
    #[diag(parse_bare_cr)]
    BareCr {
        #[primary_span]
        #[suggestion(
            parse_escape,
            applicability = "machine-applicable",
            code = "\\r",
            style = "verbose"
        )]
        span: Span,
        double_quotes: bool,
    },
    #[diag(parse_bare_cr_in_raw_string)]
    BareCrRawString(#[primary_span] Span),
    #[diag(parse_too_short_hex_escape)]
    TooShortHexEscape(#[primary_span] Span),
    #[diag(parse_invalid_char_in_escape)]
    InvalidCharInEscape {
        #[primary_span]
        #[label]
        span: Span,
        is_hex: bool,
        ch: String,
    },
    #[diag(parse_out_of_range_hex_escape)]
    OutOfRangeHexEscape(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_leading_underscore_unicode_escape)]
    LeadingUnderscoreUnicodeEscape {
        #[primary_span]
        #[label(parse_leading_underscore_unicode_escape_label)]
        span: Span,
        ch: String,
    },
    #[diag(parse_overlong_unicode_escape)]
    OverlongUnicodeEscape(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_unclosed_unicode_escape)]
    UnclosedUnicodeEscape(
        #[primary_span]
        #[label]
        Span,
        #[suggestion(
            parse_terminate,
            code = "}}",
            applicability = "maybe-incorrect",
            style = "verbose"
        )]
        Span,
    ),
    #[diag(parse_no_brace_unicode_escape)]
    NoBraceInUnicodeEscape {
        #[primary_span]
        span: Span,
        #[label]
        label: Option<Span>,
        #[subdiagnostic]
        sub: NoBraceUnicodeSub,
    },
    #[diag(parse_unicode_escape_in_byte)]
    #[help]
    UnicodeEscapeInByte(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_empty_unicode_escape)]
    EmptyUnicodeEscape(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_zero_chars)]
    ZeroChars(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_lone_slash)]
    LoneSlash(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_unskipped_whitespace)]
    UnskippedWhitespace {
        #[primary_span]
        span: Span,
        #[label]
        char_span: Span,
        ch: String,
    },
    #[diag(parse_multiple_skipped_lines)]
    MultipleSkippedLinesWarning(
        #[primary_span]
        #[label]
        Span,
    ),
    #[diag(parse_more_than_one_char)]
    MoreThanOneChar {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        note: Option<MoreThanOneCharNote>,
        #[subdiagnostic]
        suggestion: MoreThanOneCharSugg,
    },
    #[diag(parse_nul_in_c_str)]
    NulInCStr {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub enum MoreThanOneCharSugg {
    #[suggestion(
        parse_consider_normalized,
        code = "{normalized}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    NormalizedForm {
        #[primary_span]
        span: Span,
        ch: String,
        normalized: String,
    },
    #[suggestion(
        parse_remove_non,
        code = "{ch}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    RemoveNonPrinting {
        #[primary_span]
        span: Span,
        ch: String,
    },
    #[suggestion(
        parse_use_double_quotes,
        code = "{sugg}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    QuotesFull {
        #[primary_span]
        span: Span,
        is_byte: bool,
        sugg: String,
    },
    #[multipart_suggestion(parse_use_double_quotes, applicability = "machine-applicable")]
    Quotes {
        #[suggestion_part(code = "{prefix}\"")]
        start: Span,
        #[suggestion_part(code = "\"")]
        end: Span,
        is_byte: bool,
        prefix: &'static str,
    },
}

#[derive(Subdiagnostic)]
pub enum MoreThanOneCharNote {
    #[note(parse_followed_by)]
    AllCombining {
        #[primary_span]
        span: Span,
        chr: String,
        len: usize,
        escaped_marks: String,
    },
    #[note(parse_non_printing)]
    NonPrinting {
        #[primary_span]
        span: Span,
        escaped: String,
    },
}

#[derive(Subdiagnostic)]
pub enum NoBraceUnicodeSub {
    #[suggestion(
        parse_use_braces,
        code = "{suggestion}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion: String,
    },
    #[help(parse_format_of_unicode)]
    Help,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_sugg_wrap_pattern_in_parens, applicability = "machine-applicable")]
pub(crate) struct WrapInParens {
    #[suggestion_part(code = "(")]
    pub(crate) lo: Span,
    #[suggestion_part(code = ")")]
    pub(crate) hi: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum TopLevelOrPatternNotAllowedSugg {
    #[suggestion(
        parse_sugg_remove_leading_vert_in_pattern,
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    RemoveLeadingVert {
        #[primary_span]
        span: Span,
    },
    WrapInParens {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        suggestion: WrapInParens,
    },
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_vert_vert_before_function_parameter)]
#[note(parse_note_pattern_alternatives_use_single_vert)]
pub(crate) struct UnexpectedVertVertBeforeFunctionParam {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_vert_vert_in_pattern)]
pub(crate) struct UnexpectedVertVertInPattern {
    #[primary_span]
    #[suggestion(code = "|", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    #[label(parse_label_while_parsing_or_pattern_here)]
    pub start: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_trailing_vert_not_allowed)]
pub(crate) struct TrailingVertNotAllowed {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    #[label(parse_label_while_parsing_or_pattern_here)]
    pub start: Option<Span>,
    pub token: Token,
    #[note(parse_note_pattern_alternatives_use_single_vert)]
    pub note_double_vert: Option<()>,
}

#[derive(Diagnostic)]
#[diag(parse_dotdotdot_rest_pattern)]
pub(crate) struct DotDotDotRestPattern {
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion(style = "verbose", code = "", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(parse_pattern_on_wrong_side_of_at)]
pub(crate) struct PatternOnWrongSideOfAt {
    #[primary_span]
    #[suggestion(code = "{whole_pat}", applicability = "machine-applicable", style = "verbose")]
    pub whole_span: Span,
    pub whole_pat: String,
    #[label(parse_label_pattern)]
    pub pattern: Span,
    #[label(parse_label_binding)]
    pub binding: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_binding_left_of_at)]
#[note]
pub(crate) struct ExpectedBindingLeftOfAt {
    #[primary_span]
    pub whole_span: Span,
    #[label(parse_label_lhs)]
    pub lhs: Span,
    #[label(parse_label_rhs)]
    pub rhs: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_ambiguous_range_pattern_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct ParenRangeSuggestion {
    #[suggestion_part(code = "(")]
    pub lo: Span,
    #[suggestion_part(code = ")")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(parse_ambiguous_range_pattern)]
pub(crate) struct AmbiguousRangePattern {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: ParenRangeSuggestion,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_lifetime_in_pattern)]
pub(crate) struct UnexpectedLifetimeInPattern {
    #[primary_span]
    pub span: Span,
    pub symbol: Symbol,
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
pub(crate) enum InvalidMutInPattern {
    #[diag(parse_mut_on_nested_ident_pattern)]
    #[note(parse_note_mut_pattern_usage)]
    NestedIdent {
        #[primary_span]
        #[suggestion(code = "{pat}", applicability = "machine-applicable", style = "verbose")]
        span: Span,
        pat: String,
    },
    #[diag(parse_mut_on_non_ident_pattern)]
    #[note(parse_note_mut_pattern_usage)]
    NonIdent {
        #[primary_span]
        #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parse_repeated_mut_in_pattern)]
pub(crate) struct RepeatedMutInPattern {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_dot_dot_dot_range_to_pattern_not_allowed)]
pub(crate) struct DotDotDotRangeToPatternNotAllowed {
    #[primary_span]
    #[suggestion(style = "verbose", code = "..=", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_enum_pattern_instead_of_identifier)]
pub(crate) struct EnumPatternInsteadOfIdentifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_dot_dot_dot_for_remaining_fields)]
pub(crate) struct DotDotDotForRemainingFields {
    #[primary_span]
    #[suggestion(code = "..", style = "verbose", applicability = "machine-applicable")]
    pub span: Span,
    pub token_str: Cow<'static, str>,
}

#[derive(Diagnostic)]
#[diag(parse_expected_comma_after_pattern_field)]
pub(crate) struct ExpectedCommaAfterPatternField {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_expr_in_pat)]
pub(crate) struct UnexpectedExpressionInPattern {
    #[primary_span]
    #[label]
    pub span: Span,
    /// Was a `RangePatternBound` expected?
    pub is_bound: bool,
    /// Was the unexpected expression a `MethodCallExpression`?
    pub is_method_call: bool,
}

#[derive(Diagnostic)]
#[diag(parse_unexpected_paren_in_range_pat)]
pub(crate) struct UnexpectedParenInRangePat {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: UnexpectedParenInRangePatSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_unexpected_paren_in_range_pat_sugg,
    applicability = "machine-applicable"
)]
pub(crate) struct UnexpectedParenInRangePatSugg {
    #[suggestion_part(code = "")]
    pub start_span: Span,
    #[suggestion_part(code = "")]
    pub end_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_return_types_use_thin_arrow)]
pub(crate) struct ReturnTypesUseThinArrow {
    #[primary_span]
    pub span: Span,
    #[suggestion(style = "verbose", code = " -> ", applicability = "machine-applicable")]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(parse_need_plus_after_trait_object_lifetime)]
pub(crate) struct NeedPlusAfterTraitObjectLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expected_mut_or_const_in_raw_pointer_type)]
pub(crate) struct ExpectedMutOrConstInRawPointerType {
    #[primary_span]
    pub span: Span,
    #[suggestion(code("mut ", "const "), applicability = "has-placeholders", style = "verbose")]
    pub after_asterisk: Span,
}

#[derive(Diagnostic)]
#[diag(parse_lifetime_after_mut)]
pub(crate) struct LifetimeAfterMut {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "&{snippet} mut", applicability = "maybe-incorrect", style = "verbose")]
    pub suggest_lifetime: Option<Span>,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag(parse_dyn_after_mut)]
pub(crate) struct DynAfterMut {
    #[primary_span]
    #[suggestion(code = "&mut dyn", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_fn_pointer_cannot_be_const)]
pub(crate) struct FnPointerCannotBeConst {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "", applicability = "maybe-incorrect", style = "verbose")]
    #[label]
    pub qualifier: Span,
}

#[derive(Diagnostic)]
#[diag(parse_fn_pointer_cannot_be_async)]
pub(crate) struct FnPointerCannotBeAsync {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "", applicability = "maybe-incorrect", style = "verbose")]
    #[label]
    pub qualifier: Span,
}

#[derive(Diagnostic)]
#[diag(parse_nested_c_variadic_type, code = E0743)]
pub(crate) struct NestedCVariadicType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_dyn_keyword)]
#[help]
pub(crate) struct InvalidDynKeyword {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
pub enum HelpUseLatestEdition {
    #[help(parse_help_set_edition_cargo)]
    #[note(parse_note_edition_guide)]
    Cargo { edition: Edition },
    #[help(parse_help_set_edition_standalone)]
    #[note(parse_note_edition_guide)]
    Standalone { edition: Edition },
}

impl HelpUseLatestEdition {
    pub fn new() -> Self {
        let edition = LATEST_STABLE_EDITION;
        if rustc_session::utils::was_invoked_from_cargo() {
            Self::Cargo { edition }
        } else {
            Self::Standalone { edition }
        }
    }
}

#[derive(Diagnostic)]
#[diag(parse_box_syntax_removed)]
pub struct BoxSyntaxRemoved<'a> {
    #[primary_span]
    #[suggestion(
        code = "Box::new({code})",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
    pub code: &'a str,
}

#[derive(Diagnostic)]
#[diag(parse_bad_return_type_notation_output)]
pub(crate) struct BadReturnTypeNotationOutput {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_bad_assoc_type_bounds)]
pub(crate) struct BadAssocTypeBounds {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_attr_after_generic)]
pub(crate) struct AttrAfterGeneric {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_attr_without_generics)]
pub(crate) struct AttrWithoutGenerics {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_where_generics)]
pub(crate) struct WhereOnGenerics {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_generics_in_path)]
pub(crate) struct GenericsInPath {
    #[primary_span]
    pub span: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(parse_lifetime_in_eq_constraint)]
#[help]
pub(crate) struct LifetimeInEqConstraint {
    #[primary_span]
    #[label]
    pub span: Span,
    pub lifetime: Ident,
    #[label(parse_context_label)]
    pub binding_label: Span,
    #[suggestion(
        parse_colon_sugg,
        style = "verbose",
        applicability = "maybe-incorrect",
        code = ": "
    )]
    pub colon_sugg: Span,
}

#[derive(Diagnostic)]
#[diag(parse_modifier_lifetime)]
pub(crate) struct ModifierLifetime {
    #[primary_span]
    #[suggestion(style = "tool-only", applicability = "maybe-incorrect", code = "")]
    pub span: Span,
    pub modifier: &'static str,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_parenthesized_lifetime_suggestion,
    applicability = "machine-applicable"
)]
pub(crate) struct RemoveParens {
    #[suggestion_part(code = "")]
    pub lo: Span,
    #[suggestion_part(code = "")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(parse_parenthesized_lifetime)]
pub(crate) struct ParenthesizedLifetime {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: RemoveParens,
}

#[derive(Diagnostic)]
#[diag(parse_underscore_literal_suffix)]
pub(crate) struct UnderscoreLiteralSuffix {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expect_label_found_ident)]
pub(crate) struct ExpectedLabelFoundIdent {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "'", applicability = "machine-applicable", style = "verbose")]
    pub start: Span,
}

#[derive(Diagnostic)]
#[diag(parse_inappropriate_default)]
#[note]
pub(crate) struct InappropriateDefault {
    #[primary_span]
    #[label]
    pub span: Span,
    pub article: &'static str,
    pub descr: &'static str,
}

#[derive(Diagnostic)]
#[diag(parse_recover_import_as_use)]
pub(crate) struct RecoverImportAsUse {
    #[primary_span]
    #[suggestion(code = "use", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    pub token_name: String,
}

#[derive(Diagnostic)]
#[diag(parse_single_colon_import_path)]
#[note]
pub(crate) struct SingleColonImportPath {
    #[primary_span]
    #[suggestion(code = "::", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_bad_item_kind)]
pub(crate) struct BadItemKind {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
    pub ctx: &'static str,
    #[help]
    pub help: Option<()>,
}

#[derive(Diagnostic)]
#[diag(parse_single_colon_struct_type)]
pub(crate) struct SingleColonStructType {
    #[primary_span]
    #[suggestion(code = "::", applicability = "maybe-incorrect", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_equals_struct_default)]
pub(crate) struct EqualsStructDefault {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_macro_rules_missing_bang)]
pub(crate) struct MacroRulesMissingBang {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "!", applicability = "machine-applicable", style = "verbose")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag(parse_macro_name_remove_bang)]
pub(crate) struct MacroNameRemoveBang {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "short")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_macro_rules_visibility)]
pub(crate) struct MacroRulesVisibility<'a> {
    #[primary_span]
    #[suggestion(code = "#[macro_export]", applicability = "maybe-incorrect", style = "verbose")]
    pub span: Span,
    pub vis: &'a str,
}

#[derive(Diagnostic)]
#[diag(parse_macro_invocation_visibility)]
#[help]
pub(crate) struct MacroInvocationVisibility<'a> {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    pub vis: &'a str,
}

#[derive(Diagnostic)]
#[diag(parse_nested_adt)]
pub(crate) struct NestedAdt<'a> {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "", applicability = "maybe-incorrect", style = "verbose")]
    pub item: Span,
    pub keyword: &'a str,
    pub kw_str: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag(parse_function_body_equals_expr)]
pub(crate) struct FunctionBodyEqualsExpr {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: FunctionBodyEqualsExprSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct FunctionBodyEqualsExprSugg {
    #[suggestion_part(code = "{{")]
    pub eq: Span,
    #[suggestion_part(code = " }}")]
    pub semi: Span,
}

#[derive(Diagnostic)]
#[diag(parse_box_not_pat)]
pub(crate) struct BoxNotPat {
    #[primary_span]
    pub span: Span,
    #[note]
    pub kw: Span,
    #[suggestion(code = "r#", applicability = "maybe-incorrect", style = "verbose")]
    pub lo: Span,
    pub descr: String,
}

#[derive(Diagnostic)]
#[diag(parse_unmatched_angle)]
pub(crate) struct UnmatchedAngle {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable", style = "verbose")]
    pub span: Span,
    pub plural: bool,
}

#[derive(Diagnostic)]
#[diag(parse_missing_plus_in_bounds)]
pub(crate) struct MissingPlusBounds {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = " +", applicability = "maybe-incorrect", style = "verbose")]
    pub hi: Span,
    pub sym: Symbol,
}

#[derive(Diagnostic)]
#[diag(parse_incorrect_parens_trait_bounds)]
pub(crate) struct IncorrectParensTraitBounds {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: IncorrectParensTraitBoundsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_incorrect_parens_trait_bounds_sugg,
    applicability = "machine-applicable"
)]
pub(crate) struct IncorrectParensTraitBoundsSugg {
    #[suggestion_part(code = " ")]
    pub wrong_span: Span,
    #[suggestion_part(code = "(")]
    pub new_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_kw_bad_case)]
pub(crate) struct KwBadCase<'a> {
    #[primary_span]
    #[suggestion(code = "{kw}", style = "verbose", applicability = "machine-applicable")]
    pub span: Span,
    pub kw: &'a str,
}

#[derive(Diagnostic)]
#[diag(parse_meta_bad_delim)]
pub(crate) struct MetaBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Diagnostic)]
#[diag(parse_cfg_attr_bad_delim)]
pub(crate) struct CfgAttrBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_meta_bad_delim_suggestion, applicability = "machine-applicable")]
pub(crate) struct MetaBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(Diagnostic)]
#[diag(parse_malformed_cfg_attr)]
#[note]
pub(crate) struct MalformedCfgAttr {
    #[primary_span]
    #[suggestion(style = "verbose", code = "{sugg}")]
    pub span: Span,
    pub sugg: &'static str,
}

#[derive(Diagnostic)]
#[diag(parse_unknown_builtin_construct)]
pub(crate) struct UnknownBuiltinConstruct {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(parse_expected_builtin_ident)]
pub(crate) struct ExpectedBuiltinIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_static_with_generics)]
pub(crate) struct StaticWithGenerics {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_where_clause_before_const_body)]
pub(crate) struct WhereClauseBeforeConstBody {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(parse_name_label)]
    pub name: Span,
    #[label(parse_body_label)]
    pub body: Span,
    #[subdiagnostic]
    pub sugg: Option<WhereClauseBeforeConstBodySugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct WhereClauseBeforeConstBodySugg {
    #[suggestion_part(code = "= {snippet} ")]
    pub left: Span,
    pub snippet: String,
    #[suggestion_part(code = "")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_generic_args_in_pat_require_turbofish_syntax)]
pub(crate) struct GenericArgsInPatRequireTurbofishSyntax {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        parse_sugg_turbofish_syntax,
        style = "verbose",
        code = "::",
        applicability = "maybe-incorrect"
    )]
    pub suggest_turbofish: Span,
}

#[derive(Diagnostic)]
#[diag(parse_transpose_dyn_or_impl)]
pub(crate) struct TransposeDynOrImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub kw: &'a str,
    #[subdiagnostic]
    pub sugg: TransposeDynOrImplSugg<'a>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parse_suggestion, applicability = "machine-applicable")]
pub(crate) struct TransposeDynOrImplSugg<'a> {
    #[suggestion_part(code = "")]
    pub removal_span: Span,
    #[suggestion_part(code = "{kw} ")]
    pub insertion_span: Span,
    pub kw: &'a str,
}

#[derive(Diagnostic)]
#[diag(parse_array_index_offset_of)]
pub(crate) struct ArrayIndexInOffsetOf(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(parse_invalid_offset_of)]
pub(crate) struct InvalidOffsetOf(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(parse_async_impl)]
pub(crate) struct AsyncImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_expr_rarrow_call)]
#[help]
pub(crate) struct ExprRArrowCall {
    #[primary_span]
    #[suggestion(style = "verbose", applicability = "machine-applicable", code = ".")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_dot_dot_range_attribute)]
pub(crate) struct DotDotRangeAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_invalid_attr_unsafe)]
#[note]
pub struct InvalidAttrUnsafe {
    #[primary_span]
    pub span: Span,
    pub name: Path,
}

#[derive(Diagnostic)]
#[diag(parse_unsafe_attr_outside_unsafe)]
pub struct UnsafeAttrOutsideUnsafe {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: UnsafeAttrOutsideUnsafeSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parse_unsafe_attr_outside_unsafe_suggestion,
    applicability = "machine-applicable"
)]
pub struct UnsafeAttrOutsideUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parse_binder_before_modifiers)]
pub struct BinderBeforeModifiers {
    #[primary_span]
    pub binder_span: Span,
    #[label]
    pub modifiers_span: Span,
}

#[derive(Diagnostic)]
#[diag(parse_binder_and_polarity)]
pub struct BinderAndPolarity {
    #[primary_span]
    pub polarity_span: Span,
    #[label]
    pub binder_span: Span,
    pub polarity: &'static str,
}

#[derive(Diagnostic)]
#[diag(parse_modifiers_and_polarity)]
pub struct PolarityAndModifiers {
    #[primary_span]
    pub polarity_span: Span,
    #[label]
    pub modifiers_span: Span,
    pub polarity: &'static str,
    pub modifiers_concatenated: String,
}
