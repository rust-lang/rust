use rustc_ast::token::Token;
use rustc_ast::Path;
use rustc_errors::{fluent, AddToDiagnostic, Applicability, EmissionGuarantee, IntoDiagnostic};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol};

use crate::parser::TokenDescription;

#[derive(Diagnostic)]
#[diag(parser::maybe_report_ambiguous_plus)]
pub(crate) struct AmbiguousPlus {
    pub sum_ty: String,
    #[primary_span]
    #[suggestion(code = "({sum_ty})")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::maybe_recover_from_bad_type_plus, code = "E0178")]
pub(crate) struct BadTypePlus {
    pub ty: String,
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: BadTypePlusSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum BadTypePlusSub {
    #[suggestion(
        parser::add_paren,
        code = "{sum_with_parens}",
        applicability = "machine-applicable"
    )]
    AddParen {
        sum_with_parens: String,
        #[primary_span]
        span: Span,
    },
    #[label(parser::forgot_paren)]
    ForgotParen {
        #[primary_span]
        span: Span,
    },
    #[label(parser::expect_path)]
    ExpectPath {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parser::maybe_recover_from_bad_qpath_stage_2)]
pub(crate) struct BadQPathStage2 {
    #[primary_span]
    #[suggestion(code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    pub ty: String,
}

#[derive(Diagnostic)]
#[diag(parser::incorrect_semicolon)]
pub(crate) struct IncorrectSemicolon<'a> {
    #[primary_span]
    #[suggestion_short(code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[help]
    pub opt_help: Option<()>,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag(parser::incorrect_use_of_await)]
pub(crate) struct IncorrectUseOfAwait {
    #[primary_span]
    #[suggestion(parser::parentheses_suggestion, code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::incorrect_use_of_await)]
pub(crate) struct IncorrectAwait {
    #[primary_span]
    pub span: Span,
    #[suggestion(parser::postfix_suggestion, code = "{expr}.await{question_mark}")]
    pub sugg_span: (Span, Applicability),
    pub expr: String,
    pub question_mark: &'static str,
}

#[derive(Diagnostic)]
#[diag(parser::in_in_typo)]
pub(crate) struct InInTypo {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_variable_declaration)]
pub(crate) struct InvalidVariableDeclaration {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: InvalidVariableDeclarationSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidVariableDeclarationSub {
    #[suggestion(
        parser::switch_mut_let_order,
        applicability = "maybe-incorrect",
        code = "let mut"
    )]
    SwitchMutLetOrder(#[primary_span] Span),
    #[suggestion(
        parser::missing_let_before_mut,
        applicability = "machine-applicable",
        code = "let mut"
    )]
    MissingLet(#[primary_span] Span),
    #[suggestion(parser::use_let_not_auto, applicability = "machine-applicable", code = "let")]
    UseLetNotAuto(#[primary_span] Span),
    #[suggestion(parser::use_let_not_var, applicability = "machine-applicable", code = "let")]
    UseLetNotVar(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::invalid_comparison_operator)]
pub(crate) struct InvalidComparisonOperator {
    #[primary_span]
    pub span: Span,
    pub invalid: String,
    #[subdiagnostic]
    pub sub: InvalidComparisonOperatorSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidComparisonOperatorSub {
    #[suggestion_short(
        parser::use_instead,
        applicability = "machine-applicable",
        code = "{correct}"
    )]
    Correctable {
        #[primary_span]
        span: Span,
        invalid: String,
        correct: String,
    },
    #[label(parser::spaceship_operator_invalid)]
    Spaceship(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::invalid_logical_operator)]
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
    #[suggestion_short(
        parser::use_amp_amp_for_conjunction,
        applicability = "machine-applicable",
        code = "&&"
    )]
    Conjunction(#[primary_span] Span),
    #[suggestion_short(
        parser::use_pipe_pipe_for_disjunction,
        applicability = "machine-applicable",
        code = "||"
    )]
    Disjunction(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::tilde_is_not_unary_operator)]
pub(crate) struct TildeAsUnaryOperator(
    #[primary_span]
    #[suggestion_short(applicability = "machine-applicable", code = "!")]
    pub Span,
);

#[derive(Diagnostic)]
#[diag(parser::unexpected_token_after_not)]
pub(crate) struct NotAsNegationOperator {
    #[primary_span]
    pub negated: Span,
    pub negated_desc: String,
    #[subdiagnostic]
    pub sub: NotAsNegationOperatorSub,
}

#[derive(Subdiagnostic)]
pub enum NotAsNegationOperatorSub {
    #[suggestion_short(
        parser::unexpected_token_after_not_default,
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotDefault(#[primary_span] Span),

    #[suggestion_short(
        parser::unexpected_token_after_not_bitwise,
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotBitwise(#[primary_span] Span),

    #[suggestion_short(
        parser::unexpected_token_after_not_logical,
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotLogical(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::malformed_loop_label)]
pub(crate) struct MalformedLoopLabel {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "{correct_label}")]
    pub span: Span,
    pub correct_label: Ident,
}

#[derive(Diagnostic)]
#[diag(parser::lifetime_in_borrow_expression)]
pub(crate) struct LifetimeInBorrowExpression {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable", code = "")]
    #[label]
    pub lifetime_span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::field_expression_with_generic)]
pub(crate) struct FieldExpressionWithGeneric(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(parser::macro_invocation_with_qualified_path)]
pub(crate) struct MacroInvocationWithQualifiedPath(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(parser::unexpected_token_after_label)]
pub(crate) struct UnexpectedTokenAfterLabel {
    #[primary_span]
    #[label(parser::unexpected_token_after_label)]
    pub span: Span,
    #[suggestion_verbose(parser::suggestion_remove_label, code = "")]
    pub remove_label: Option<Span>,
    #[subdiagnostic]
    pub enclose_in_block: Option<UnexpectedTokenAfterLabelSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion_enclose_in_block, applicability = "machine-applicable")]
pub(crate) struct UnexpectedTokenAfterLabelSugg {
    #[suggestion_part(code = "{{ ")]
    pub left: Span,
    #[suggestion_part(code = " }}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::require_colon_after_labeled_expression)]
#[note]
pub(crate) struct RequireColonAfterLabeledExpression {
    #[primary_span]
    pub span: Span,
    #[label]
    pub label: Span,
    #[suggestion_short(applicability = "machine-applicable", code = ": ")]
    pub label_end: Span,
}

#[derive(Diagnostic)]
#[diag(parser::do_catch_syntax_removed)]
#[note]
pub(crate) struct DoCatchSyntaxRemoved {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "try")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::float_literal_requires_integer_part)]
pub(crate) struct FloatLiteralRequiresIntegerPart {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "{correct}")]
    pub span: Span,
    pub correct: String,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_int_literal_width)]
#[help]
pub(crate) struct InvalidIntLiteralWidth {
    #[primary_span]
    pub span: Span,
    pub width: String,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_num_literal_base_prefix)]
#[note]
pub(crate) struct InvalidNumLiteralBasePrefix {
    #[primary_span]
    #[suggestion(applicability = "maybe-incorrect", code = "{fixed}")]
    pub span: Span,
    pub fixed: String,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_num_literal_suffix)]
#[help]
pub(crate) struct InvalidNumLiteralSuffix {
    #[primary_span]
    #[label]
    pub span: Span,
    pub suffix: String,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_float_literal_width)]
#[help]
pub(crate) struct InvalidFloatLiteralWidth {
    #[primary_span]
    pub span: Span,
    pub width: String,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_float_literal_suffix)]
#[help]
pub(crate) struct InvalidFloatLiteralSuffix {
    #[primary_span]
    #[label]
    pub span: Span,
    pub suffix: String,
}

#[derive(Diagnostic)]
#[diag(parser::int_literal_too_large)]
pub(crate) struct IntLiteralTooLarge {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::missing_semicolon_before_array)]
pub(crate) struct MissingSemicolonBeforeArray {
    #[primary_span]
    pub open_delim: Span,
    #[suggestion_verbose(applicability = "maybe-incorrect", code = ";")]
    pub semicolon: Span,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_block_macro_segment)]
pub(crate) struct InvalidBlockMacroSegment {
    #[primary_span]
    pub span: Span,
    #[label]
    pub context: Span,
}

#[derive(Diagnostic)]
#[diag(parser::if_expression_missing_then_block)]
pub(crate) struct IfExpressionMissingThenBlock {
    #[primary_span]
    pub if_span: Span,
    #[subdiagnostic]
    pub sub: IfExpressionMissingThenBlockSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum IfExpressionMissingThenBlockSub {
    #[help(parser::condition_possibly_unfinished)]
    UnfinishedCondition(#[primary_span] Span),
    #[help(parser::add_then_block)]
    AddThenBlock(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::if_expression_missing_condition)]
pub(crate) struct IfExpressionMissingCondition {
    #[primary_span]
    #[label(parser::condition_label)]
    pub if_span: Span,
    #[label(parser::block_label)]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::expected_expression_found_let)]
pub(crate) struct ExpectedExpressionFoundLet {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::expected_else_block)]
pub(crate) struct ExpectedElseBlock {
    #[primary_span]
    pub first_tok_span: Span,
    pub first_tok: String,
    #[label]
    pub else_span: Span,
    #[suggestion(applicability = "maybe-incorrect", code = "if ")]
    pub condition_start: Span,
}

#[derive(Diagnostic)]
#[diag(parser::outer_attribute_not_allowed_on_if_else)]
pub(crate) struct OuterAttributeNotAllowedOnIfElse {
    #[primary_span]
    pub last: Span,

    #[label(parser::branch_label)]
    pub branch_span: Span,

    #[label(parser::ctx_label)]
    pub ctx_span: Span,
    pub ctx: String,

    #[suggestion(applicability = "machine-applicable", code = "")]
    pub attributes: Span,
}

#[derive(Diagnostic)]
#[diag(parser::missing_in_in_for_loop)]
pub(crate) struct MissingInInForLoop {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: MissingInInForLoopSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum MissingInInForLoopSub {
    // Has been misleading, at least in the past (closed Issue #48492), thus maybe-incorrect
    #[suggestion_short(parser::use_in_not_of, applicability = "maybe-incorrect", code = "in")]
    InNotOf(#[primary_span] Span),
    #[suggestion_short(parser::add_in, applicability = "maybe-incorrect", code = " in ")]
    AddIn(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::missing_comma_after_match_arm)]
pub(crate) struct MissingCommaAfterMatchArm {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = ",")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::catch_after_try)]
#[help]
pub(crate) struct CatchAfterTry {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::comma_after_base_struct)]
#[note]
pub(crate) struct CommaAfterBaseStruct {
    #[primary_span]
    pub span: Span,
    #[suggestion_short(applicability = "machine-applicable", code = "")]
    pub comma: Span,
}

#[derive(Diagnostic)]
#[diag(parser::eq_field_init)]
pub(crate) struct EqFieldInit {
    #[primary_span]
    pub span: Span,
    #[suggestion(applicability = "machine-applicable", code = ":")]
    pub eq: Span,
}

#[derive(Diagnostic)]
#[diag(parser::dotdotdot)]
pub(crate) struct DotDotDot {
    #[primary_span]
    #[suggestion(parser::suggest_exclusive_range, applicability = "maybe-incorrect", code = "..")]
    #[suggestion(parser::suggest_inclusive_range, applicability = "maybe-incorrect", code = "..=")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::left_arrow_operator)]
pub(crate) struct LeftArrowOperator {
    #[primary_span]
    #[suggestion(applicability = "maybe-incorrect", code = "< -")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::remove_let)]
pub(crate) struct RemoveLet {
    #[primary_span]
    #[suggestion(applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::use_eq_instead)]
pub(crate) struct UseEqInstead {
    #[primary_span]
    #[suggestion_short(applicability = "machine-applicable", code = "=")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::use_empty_block_not_semi)]
pub(crate) struct UseEmptyBlockNotSemi {
    #[primary_span]
    #[suggestion_hidden(applicability = "machine-applicable", code = "{{}}")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::comparison_interpreted_as_generic)]
pub(crate) struct ComparisonInterpretedAsGeneric {
    #[primary_span]
    #[label(parser::label_comparison)]
    pub comparison: Span,
    pub r#type: Path,
    #[label(parser::label_args)]
    pub args: Span,
    #[subdiagnostic]
    pub suggestion: ComparisonOrShiftInterpretedAsGenericSugg,
}

#[derive(Diagnostic)]
#[diag(parser::shift_interpreted_as_generic)]
pub(crate) struct ShiftInterpretedAsGeneric {
    #[primary_span]
    #[label(parser::label_comparison)]
    pub shift: Span,
    pub r#type: Path,
    #[label(parser::label_args)]
    pub args: Span,
    #[subdiagnostic]
    pub suggestion: ComparisonOrShiftInterpretedAsGenericSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
pub(crate) struct ComparisonOrShiftInterpretedAsGenericSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::found_expr_would_be_stmt)]
pub(crate) struct FoundExprWouldBeStmt {
    #[primary_span]
    #[label]
    pub span: Span,
    pub token: Token,
    #[subdiagnostic]
    pub suggestion: ExprParenthesesNeeded,
}

#[derive(Diagnostic)]
#[diag(parser::leading_plus_not_supported)]
pub(crate) struct LeadingPlusNotSupported {
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion_verbose(
        parser::suggestion_remove_plus,
        code = "",
        applicability = "machine-applicable"
    )]
    pub remove_plus: Option<Span>,
    #[subdiagnostic]
    pub add_parentheses: Option<ExprParenthesesNeeded>,
}

#[derive(Diagnostic)]
#[diag(parser::parentheses_with_struct_fields)]
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
#[multipart_suggestion(parser::suggestion_braces_for_struct, applicability = "maybe-incorrect")]
pub(crate) struct BracesForStructLiteral {
    #[suggestion_part(code = " {{ ")]
    pub first: Span,
    #[suggestion_part(code = " }}")]
    pub second: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion_no_fields_for_fn, applicability = "maybe-incorrect")]
pub(crate) struct NoFieldsForFnCall {
    #[suggestion_part(code = "")]
    pub fields: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(parser::labeled_loop_in_break)]
pub(crate) struct LabeledLoopInBreak {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: WrapExpressionInParentheses,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    parser::sugg_wrap_expression_in_parentheses,
    applicability = "machine-applicable"
)]
pub(crate) struct WrapExpressionInParentheses {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::array_brackets_instead_of_braces)]
pub(crate) struct ArrayBracketsInsteadOfSpaces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: ArrayBracketsInsteadOfSpacesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "maybe-incorrect")]
pub(crate) struct ArrayBracketsInsteadOfSpacesSugg {
    #[suggestion_part(code = "[")]
    pub left: Span,
    #[suggestion_part(code = "]")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::match_arm_body_without_braces)]
pub(crate) struct MatchArmBodyWithoutBraces {
    #[primary_span]
    #[label(parser::label_statements)]
    pub statements: Span,
    #[label(parser::label_arrow)]
    pub arrow: Span,
    pub num_statements: usize,
    #[subdiagnostic]
    pub sub: MatchArmBodyWithoutBracesSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum MatchArmBodyWithoutBracesSugg {
    #[multipart_suggestion(parser::suggestion_add_braces, applicability = "machine-applicable")]
    AddBraces {
        #[suggestion_part(code = "{{ ")]
        left: Span,
        #[suggestion_part(code = " }}")]
        right: Span,
    },
    #[suggestion(
        parser::suggestion_use_comma_not_semicolon,
        code = ",",
        applicability = "machine-applicable"
    )]
    UseComma {
        #[primary_span]
        semicolon: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parser::struct_literal_not_allowed_here)]
pub(crate) struct StructLiteralNotAllowedHere {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: StructLiteralNotAllowedHereSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
pub(crate) struct StructLiteralNotAllowedHereSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_interpolated_expression)]
pub(crate) struct InvalidInterpolatedExpression {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::hexadecimal_float_literal_not_supported)]
pub(crate) struct HexadecimalFloatLiteralNotSupported {
    #[primary_span]
    #[label(parser::not_supported)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::octal_float_literal_not_supported)]
pub(crate) struct OctalFloatLiteralNotSupported {
    #[primary_span]
    #[label(parser::not_supported)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::binary_float_literal_not_supported)]
pub(crate) struct BinaryFloatLiteralNotSupported {
    #[primary_span]
    #[label(parser::not_supported)]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_literal_suffix)]
pub(crate) struct InvalidLiteralSuffix {
    #[primary_span]
    #[label]
    pub span: Span,
    // FIXME(#100717)
    pub kind: String,
    pub suffix: Symbol,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_literal_suffix_on_tuple_index)]
pub(crate) struct InvalidLiteralSuffixOnTupleIndex {
    #[primary_span]
    #[label]
    pub span: Span,
    pub suffix: Symbol,
    #[help(parser::tuple_exception_line_1)]
    #[help(parser::tuple_exception_line_2)]
    #[help(parser::tuple_exception_line_3)]
    pub exception: Option<()>,
}

#[derive(Diagnostic)]
#[diag(parser::non_string_abi_literal)]
pub(crate) struct NonStringAbiLiteral {
    #[primary_span]
    #[suggestion(code = "\"C\"", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::mismatched_closing_delimiter)]
pub(crate) struct MismatchedClosingDelimiter {
    #[primary_span]
    pub spans: Vec<Span>,
    pub delimiter: String,
    #[label(parser::label_unmatched)]
    pub unmatched: Span,
    #[label(parser::label_opening_candidate)]
    pub opening_candidate: Option<Span>,
    #[label(parser::label_unclosed)]
    pub unclosed: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parser::incorrect_visibility_restriction, code = "E0704")]
#[help]
pub(crate) struct IncorrectVisibilityRestriction {
    #[primary_span]
    #[suggestion(code = "in {inner_str}", applicability = "machine-applicable")]
    pub span: Span,
    pub inner_str: String,
}

#[derive(Diagnostic)]
#[diag(parser::assignment_else_not_allowed)]
pub(crate) struct AssignmentElseNotAllowed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::expected_statement_after_outer_attr)]
pub(crate) struct ExpectedStatementAfterOuterAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::doc_comment_does_not_document_anything, code = "E0585")]
#[help]
pub(crate) struct DocCommentDoesNotDocumentAnything {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = ",", applicability = "machine-applicable")]
    pub missing_comma: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parser::const_let_mutually_exclusive)]
pub(crate) struct ConstLetMutuallyExclusive {
    #[primary_span]
    #[suggestion(code = "const", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_expression_in_let_else)]
pub(crate) struct InvalidExpressionInLetElse {
    #[primary_span]
    pub span: Span,
    pub operator: &'static str,
    #[subdiagnostic]
    pub sugg: WrapExpressionInParentheses,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_curly_in_let_else)]
pub(crate) struct InvalidCurlyInLetElse {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WrapExpressionInParentheses,
}

#[derive(Diagnostic)]
#[diag(parser::compound_assignment_expression_in_let)]
#[help]
pub(crate) struct CompoundAssignmentExpressionInLet {
    #[primary_span]
    #[suggestion_short(code = "=", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::suffixed_literal_in_attribute)]
#[help]
pub(crate) struct SuffixedLiteralInAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::invalid_meta_item)]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub token: Token,
}

#[derive(Subdiagnostic)]
#[suggestion_verbose(
    parser::sugg_escape_to_use_as_identifier,
    applicability = "maybe-incorrect",
    code = "r#"
)]
pub(crate) struct SuggEscapeToUseAsIdentifier {
    #[primary_span]
    pub span: Span,
    pub ident_name: String,
}

#[derive(Subdiagnostic)]
#[suggestion(parser::sugg_remove_comma, applicability = "machine-applicable", code = "")]
pub(crate) struct SuggRemoveComma {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedIdentifierFound {
    #[label(parser::expected_identifier_found_reserved_identifier)]
    ReservedIdentifier(#[primary_span] Span),
    #[label(parser::expected_identifier_found_keyword)]
    Keyword(#[primary_span] Span),
    #[label(parser::expected_identifier_found_reserved_keyword)]
    ReservedKeyword(#[primary_span] Span),
    #[label(parser::expected_identifier_found_doc_comment)]
    DocComment(#[primary_span] Span),
    #[label(parser::expected_identifier)]
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
    pub suggest_raw: Option<SuggEscapeToUseAsIdentifier>,
    pub suggest_remove_comma: Option<SuggRemoveComma>,
}

impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for ExpectedIdentifier {
    fn into_diagnostic(
        self,
        handler: &'a rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'a, G> {
        let token_descr = super::parser::TokenDescription::from_token(&self.token);

        let mut diag = handler.struct_diagnostic(match token_descr {
            Some(TokenDescription::ReservedIdentifier) => {
                fluent::parser::expected_identifier_found_reserved_identifier_str
            }
            Some(TokenDescription::Keyword) => {
                fluent::parser::expected_identifier_found_keyword_str
            }
            Some(TokenDescription::ReservedKeyword) => {
                fluent::parser::expected_identifier_found_reserved_keyword_str
            }
            Some(TokenDescription::DocComment) => {
                fluent::parser::expected_identifier_found_doc_comment_str
            }
            None => fluent::parser::expected_identifier_found_str,
        });
        diag.set_span(self.span);
        diag.set_arg("token", self.token);

        if let Some(sugg) = self.suggest_raw {
            sugg.add_to_diagnostic(&mut diag);
        }

        ExpectedIdentifierFound::new(token_descr, self.span).add_to_diagnostic(&mut diag);

        if let Some(sugg) = self.suggest_remove_comma {
            sugg.add_to_diagnostic(&mut diag);
        }

        diag
    }
}

pub(crate) struct ExpectedSemi {
    pub span: Span,
    pub token: Token,

    pub unexpected_token_label: Option<Span>,
    pub sugg: ExpectedSemiSugg,
}

impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for ExpectedSemi {
    fn into_diagnostic(
        self,
        handler: &'a rustc_errors::Handler,
    ) -> rustc_errors::DiagnosticBuilder<'a, G> {
        let token_descr = super::parser::TokenDescription::from_token(&self.token);

        let mut diag = handler.struct_diagnostic(match token_descr {
            Some(TokenDescription::ReservedIdentifier) => {
                fluent::parser::expected_semi_found_reserved_identifier_str
            }
            Some(TokenDescription::Keyword) => fluent::parser::expected_semi_found_keyword_str,
            Some(TokenDescription::ReservedKeyword) => {
                fluent::parser::expected_semi_found_reserved_keyword_str
            }
            Some(TokenDescription::DocComment) => {
                fluent::parser::expected_semi_found_doc_comment_str
            }
            None => fluent::parser::expected_semi_found_str,
        });
        diag.set_span(self.span);
        diag.set_arg("token", self.token);

        if let Some(unexpected_token_label) = self.unexpected_token_label {
            diag.span_label(unexpected_token_label, fluent::parser::label_unexpected_token);
        }

        self.sugg.add_to_diagnostic(&mut diag);

        diag
    }
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedSemiSugg {
    #[suggestion(
        parser::sugg_change_this_to_semi,
        code = ";",
        applicability = "machine-applicable"
    )]
    ChangeToSemi(#[primary_span] Span),
    #[suggestion_short(parser::sugg_add_semi, code = ";", applicability = "machine-applicable")]
    AddSemi(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(parser::struct_literal_body_without_path)]
pub(crate) struct StructLiteralBodyWithoutPath {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: StructLiteralBodyWithoutPathSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "has-placeholders")]
pub(crate) struct StructLiteralBodyWithoutPathSugg {
    #[suggestion_part(code = "{{ SomeStruct ")]
    pub before: Span,
    #[suggestion_part(code = " }}")]
    pub after: Span,
}

#[derive(Diagnostic)]
#[diag(parser::unmatched_angle_brackets)]
pub(crate) struct UnmatchedAngleBrackets {
    #[primary_span]
    #[suggestion(code = "", applicability = "machine-applicable")]
    pub span: Span,
    pub num_extra_brackets: usize,
}

#[derive(Diagnostic)]
#[diag(parser::generic_parameters_without_angle_brackets)]
pub(crate) struct GenericParamsWithoutAngleBrackets {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: GenericParamsWithoutAngleBracketsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
pub(crate) struct GenericParamsWithoutAngleBracketsSugg {
    #[suggestion_part(code = "<")]
    pub left: Span,
    #[suggestion_part(code = ">")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::comparison_operators_cannot_be_chained)]
pub(crate) struct ComparisonOperatorsCannotBeChained {
    #[primary_span]
    pub span: Vec<Span>,
    #[suggestion_verbose(
        parser::sugg_turbofish_syntax,
        code = "::",
        applicability = "maybe-incorrect"
    )]
    pub suggest_turbofish: Option<Span>,
    #[help(parser::sugg_turbofish_syntax)]
    #[help(parser::sugg_parentheses_for_function_args)]
    pub help_turbofish: Option<()>,
    #[subdiagnostic]
    pub chaining_sugg: Option<ComparisonOperatorsCannotBeChainedSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ComparisonOperatorsCannotBeChainedSugg {
    #[suggestion_verbose(
        parser::sugg_split_comparison,
        code = " && {middle_term}",
        applicability = "maybe-incorrect"
    )]
    SplitComparison {
        #[primary_span]
        span: Span,
        middle_term: String,
    },
    #[multipart_suggestion(parser::sugg_parenthesize, applicability = "maybe-incorrect")]
    Parenthesize {
        #[suggestion_part(code = "(")]
        left: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
}

#[derive(Diagnostic)]
#[diag(parser::question_mark_in_type)]
pub(crate) struct QuestionMarkInType {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: QuestionMarkInTypeSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
pub(crate) struct QuestionMarkInTypeSugg {
    #[suggestion_part(code = "Option<")]
    pub left: Span,
    #[suggestion_part(code = ">")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::unexpected_parentheses_in_for_head)]
pub(crate) struct ParenthesesInForHead {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: ParenthesesInForHeadSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
pub(crate) struct ParenthesesInForHeadSugg {
    #[suggestion_part(code = "")]
    pub left: Span,
    #[suggestion_part(code = "")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::doc_comment_on_param_type)]
pub(crate) struct DocCommentOnParamType {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::attribute_on_param_type)]
pub(crate) struct AttributeOnParamType {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::pattern_method_param_without_body, code = "E0642")]
pub(crate) struct PatternMethodParamWithoutBody {
    #[primary_span]
    #[suggestion(code = "_", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::self_param_not_first)]
pub(crate) struct SelfParamNotFirst {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::const_generic_without_braces)]
pub(crate) struct ConstGenericWithoutBraces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ConstGenericWithoutBracesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
pub(crate) struct ConstGenericWithoutBracesSugg {
    #[suggestion_part(code = "{{ ")]
    pub left: Span,
    #[suggestion_part(code = " }}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(parser::unexpected_const_param_declaration)]
pub(crate) struct UnexpectedConstParamDeclaration {
    #[primary_span]
    #[label]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<UnexpectedConstParamDeclarationSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnexpectedConstParamDeclarationSugg {
    #[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
    AddParam {
        #[suggestion_part(code = "<{snippet}>")]
        impl_generics: Span,
        #[suggestion_part(code = "{ident}")]
        incorrect_decl: Span,
        snippet: String,
        ident: String,
    },
    #[multipart_suggestion(parser::suggestion, applicability = "machine-applicable")]
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
#[diag(parser::unexpected_const_in_generic_param)]
pub(crate) struct UnexpectedConstInGenericParam {
    #[primary_span]
    pub span: Span,
    #[suggestion_verbose(code = "", applicability = "maybe-incorrect")]
    pub to_remove: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(parser::async_move_order_incorrect)]
pub(crate) struct AsyncMoveOrderIncorrect {
    #[primary_span]
    #[suggestion_verbose(code = "async move", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(parser::double_colon_in_bound)]
pub(crate) struct DoubleColonInBound {
    #[primary_span]
    pub span: Span,
    #[suggestion(code = ": ", applicability = "machine-applicable")]
    pub between: Span,
}
