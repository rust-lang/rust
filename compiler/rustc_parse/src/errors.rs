// ignore-tidy-filelength
use std::borrow::Cow;

use rustc_ast::token::Token;
use rustc_ast::{Path, Visibility};
use rustc_errors::{
    AddToDiagnostic, Applicability, DiagCtxt, DiagnosticBuilder, IntoDiagnostic, Level,
    SubdiagnosticMessage, DiagnosticMessage
};
use rustc_errors::{AddToDiagnostic, Applicability, ErrorGuaranteed, IntoDiagnostic};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::edition::{Edition, LATEST_STABLE_EDITION};
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol};

use crate::fluent_generated as fluent;
use crate::parser::{ForbiddenLetReason, TokenDescription};

#[derive(Diagnostic)]
#[diag("ambiguous `+` in a type")]
pub(crate) struct AmbiguousPlus {
    pub sum_ty: String,
    #[primary_span]
    #[suggestion(label = "use parentheses to disambiguate", code = "({sum_ty})")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(label = "expected a path on the left-hand side of `+`, not `{$ty}`", code = "E0178")]
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
        label = "try adding parentheses",
        code = "{sum_with_parens}",
        applicability = "machine-applicable"
    )]
    AddParen {
        sum_with_parens: String,
        #[primary_span]
        span: Span,
    },
    #[label("perhaps you forgot parentheses?")]
    ForgotParen {
        #[primary_span]
        span: Span,
    },
    #[label("expected a path")]
    ExpectPath {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("missing angle brackets in associated item path")]
pub(crate) struct BadQPathStage2 {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub wrap: WrapType,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "types that don't start with an identifier need to be surrounded with angle brackets in qualified paths",
    applicability = "machine-applicable"
)]
pub(crate) struct WrapType {
    #[suggestion_part(code = "<")]
    pub lo: Span,
    #[suggestion_part(code = ">")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("expected item, found `;`")]
pub(crate) struct IncorrectSemicolon<'a> {
    #[primary_span]
    #[suggestion(
        label = "remove this semicolon",
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    #[help("{$name} declarations are not followed by a semicolon")]
    pub opt_help: Option<()>,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag("incorrect use of `await`")]
pub(crate) struct IncorrectUseOfAwait {
    #[primary_span]
    #[suggestion(
        label = "`await` is not a method call, remove the parentheses",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("incorrect use of `await`")]
pub(crate) struct IncorrectAwait {
    #[primary_span]
    pub span: Span,
    #[suggestion(label = "`await` is a postfix operation", code = "{expr}.await{question_mark}")]
    pub sugg_span: (Span, Applicability),
    pub expr: String,
    pub question_mark: &'static str,
}

#[derive(Diagnostic)]
#[diag("expected iterable, found keyword `in`")]
pub(crate) struct InInTypo {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "remove the duplicated `in`",
        code = "",
        applicability = "machine-applicable"
    )]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid variable declaration")]
pub(crate) struct InvalidVariableDeclaration {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: InvalidVariableDeclarationSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidVariableDeclarationSub {
    #[suggestion(
        label = "switch the order of `mut` and `let`",
        applicability = "maybe-incorrect",
        code = "let mut"
    )]
    SwitchMutLetOrder(#[primary_span] Span),
    #[suggestion(
        label = "missing keyword",
        applicability = "machine-applicable",
        code = "let mut"
    )]
    MissingLet(#[primary_span] Span),
    #[suggestion(
        label = "write `let` instead of `auto` to introduce a new variable",
        applicability = "machine-applicable",
        code = "let"
    )]
    UseLetNotAuto(#[primary_span] Span),
    #[suggestion(
        label = "write `let` instead of `var` to introduce a new variable",
        applicability = "machine-applicable",
        code = "let"
    )]
    UseLetNotVar(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("switch the order of `ref` and `box`")]
pub(crate) struct SwitchRefBoxOrder {
    #[primary_span]
    #[suggestion(label = "swap them", applicability = "machine-applicable", code = "box ref")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid comparison operator `{$invalid}`")]
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
        label = "`{$invalid}` is not a valid comparison operator, use `{$correct}`",
        style = "short",
        applicability = "machine-applicable",
        code = "{correct}"
    )]
    Correctable {
        #[primary_span]
        span: Span,
        invalid: String,
        correct: String,
    },
    #[label("`<=>` is not a valid comparison operator, use `std::cmp::Ordering`")]
    Spaceship(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("`{$incorrect}` is not a logical operator")]
#[note("unlike in e.g., Python and PHP, `&&` and `||` are used for logical operators")]
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
        label = "use `&&` to perform logical conjunction",
        style = "short",
        applicability = "machine-applicable",
        code = "&&"
    )]
    Conjunction(#[primary_span] Span),
    #[suggestion(
        label = "use `||` to perform logical disjunction",
        style = "short",
        applicability = "machine-applicable",
        code = "||"
    )]
    Disjunction(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("`~` cannot be used as a unary operator")]
pub(crate) struct TildeAsUnaryOperator(
    #[primary_span]
    #[suggestion(
        label = "use `!` to perform bitwise not",
        style = "short",
        applicability = "machine-applicable",
        code = "!"
    )]
    pub Span,
);

#[derive(Diagnostic)]
#[diag("unexpected {$negated_desc} after identifier")]
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
        label = "use `!` to perform logical negation or bitwise not",
        style = "short",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotDefault(#[primary_span] Span),

    #[suggestion(
        label = "use `!` to perform bitwise not",
        style = "short",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotBitwise(#[primary_span] Span),

    #[suggestion(
        label = "use `!` to perform logical negation",
        style = "short",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotLogical(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("malformed loop label")]
pub(crate) struct MalformedLoopLabel {
    #[primary_span]
    #[suggestion(
        label = "use the correct loop label format",
        applicability = "machine-applicable",
        code = "{correct_label}"
    )]
    pub span: Span,
    pub correct_label: Ident,
}

#[derive(Diagnostic)]
#[diag("borrow expressions cannot be annotated with lifetimes")]
pub(crate) struct LifetimeInBorrowExpression {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "remove the lifetime annotation",
        applicability = "machine-applicable",
        code = ""
    )]
    #[label("annotated with lifetime here")]
    pub lifetime_span: Span,
}

#[derive(Diagnostic)]
#[diag("field expressions cannot have generic arguments")]
pub(crate) struct FieldExpressionWithGeneric(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag("macros cannot use qualified paths")]
pub(crate) struct MacroInvocationWithQualifiedPath(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag("expected `while`, `for`, `loop` or `{` after a label")]
pub(crate) struct UnexpectedTokenAfterLabel {
    #[primary_span]
    #[label("expected `while`, `for`, `loop` or `{` after a label")]
    pub span: Span,
    #[suggestion(label = "consider removing the label", style = "verbose", code = "")]
    pub remove_label: Option<Span>,
    #[subdiagnostic]
    pub enclose_in_block: Option<UnexpectedTokenAfterLabelSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "consider enclosing expression in a block",
    applicability = "machine-applicable"
)]
pub(crate) struct UnexpectedTokenAfterLabelSugg {
    #[suggestion_part(code = "{{ ")]
    pub left: Span,
    #[suggestion_part(code = " }}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("labeled expression must be followed by `:`")]
#[note("labels are used before loops and blocks, allowing e.g., `break 'label` to them")]
pub(crate) struct RequireColonAfterLabeledExpression {
    #[primary_span]
    pub span: Span,
    #[label("the label")]
    pub label: Span,
    #[suggestion(
        label = "add `:` after the label",
        style = "short",
        applicability = "machine-applicable",
        code = ": "
    )]
    pub label_end: Span,
}

#[derive(Diagnostic)]
#[diag("found removed `do catch` syntax")]
#[note("following RFC #2388, the new non-placeholder syntax is `try`")]
pub(crate) struct DoCatchSyntaxRemoved {
    #[primary_span]
    #[suggestion(
        label = "replace with the new syntax",
        applicability = "machine-applicable",
        code = "try"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("float literals must have an integer part")]
pub(crate) struct FloatLiteralRequiresIntegerPart {
    #[primary_span]
    #[suggestion(
        label = "must have an integer part",
        applicability = "machine-applicable",
        code = "{correct}"
    )]
    pub span: Span,
    pub correct: String,
}

#[derive(Diagnostic)]
#[diag("expected `;`, found `[`")]
pub(crate) struct MissingSemicolonBeforeArray {
    #[primary_span]
    pub open_delim: Span,
    #[suggestion(
        label = "consider adding `;` here",
        style = "verbose",
        applicability = "maybe-incorrect",
        code = ";"
    )]
    pub semicolon: Span,
}

#[derive(Diagnostic)]
#[diag("expected `..`, found `...`")]
pub(crate) struct MissingDotDot {
    #[primary_span]
    pub token_span: Span,
    #[suggestion(
        label = "use `..` to fill in the rest of the fields",
        applicability = "maybe-incorrect",
        code = "..",
        style = "verbose"
    )]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot use a `block` macro fragment here")]
pub(crate) struct InvalidBlockMacroSegment {
    #[primary_span]
    pub span: Span,
    #[label("the `block` fragment is within this context")]
    pub context: Span,
    #[subdiagnostic]
    pub wrap: WrapInExplicitBlock,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(label = "wrap this in another block", applicability = "machine-applicable")]
pub(crate) struct WrapInExplicitBlock {
    #[suggestion_part(code = "{{ ")]
    pub lo: Span,
    #[suggestion_part(code = " }}")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("this `if` expression is missing a block after the condition")]
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
    #[help("this binary operation is possibly unfinished")]
    UnfinishedCondition(#[primary_span] Span),
    #[help("add a block here")]
    AddThenBlock(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("Rust has no ternary operator")]
#[help("use an `if-else` expression instead")]
pub struct TernaryOperator {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    label = "remove the `if` if you meant to write a `let...else` statement",
    applicability = "maybe-incorrect",
    code = ""
)]
pub(crate) struct IfExpressionLetSomeSub {
    #[primary_span]
    pub if_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing condition for `if` expression")]
pub(crate) struct IfExpressionMissingCondition {
    #[primary_span]
    #[label("expected condition here")]
    pub if_span: Span,
    #[label(
        "if this block is the condition of the `if` expression, then it must be followed by another block"
    )]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag("expected expression, found `let` statement")]
#[note("only supported directly in conditions of `if` and `while` expressions")]
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
#[diag("expected `=`, found `==`")]
pub(crate) struct ExpectedEqForLetExpr {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "consider using `=` here",
        applicability = "maybe-incorrect",
        code = "=",
        style = "verbose"
    )]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag(label = r#"expected `{"{"}`, found {$first_tok}"#)]
pub(crate) struct ExpectedElseBlock {
    #[primary_span]
    pub first_tok_span: Span,
    pub first_tok: String,
    #[label("expected an `if` or a block after this `else`")]
    pub else_span: Span,
    #[suggestion(
        label = "add an `if` if this is the condition of a chained `else if` statement",
        applicability = "maybe-incorrect",
        code = "if "
    )]
    pub condition_start: Span,
}

#[derive(Diagnostic)]
#[diag(label = r#"expected one of `,`, `:`, or `{"}"}`, found `{$token}`"#)]
pub(crate) struct ExpectedStructField {
    #[primary_span]
    #[label("expected one of `,`, `:`, or `}`")]
    pub span: Span,
    pub token: Token,
    #[label("while parsing this struct field")]
    pub ident_span: Span,
}

#[derive(Diagnostic)]
#[diag("outer attributes are not allowed on `if` and `else` branches")]
pub(crate) struct OuterAttributeNotAllowedOnIfElse {
    #[primary_span]
    pub last: Span,

    #[label("the attributes are attached to this branch")]
    pub branch_span: Span,

    #[label("the branch belongs to this `{$ctx}`")]
    pub ctx_span: Span,
    pub ctx: String,

    #[suggestion(label = "remove the attributes", applicability = "machine-applicable", code = "")]
    pub attributes: Span,
}

#[derive(Diagnostic)]
#[diag("missing `in` in `for` loop")]
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
        label = "try using `in` here instead",
        style = "short",
        applicability = "maybe-incorrect",
        code = "in"
    )]
    InNotOf(#[primary_span] Span),
    #[suggestion(
        label = "try adding `in` here",
        style = "short",
        applicability = "maybe-incorrect",
        code = " in "
    )]
    AddIn(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("missing expression to iterate on in `for` loop")]
pub(crate) struct MissingExpressionInForLoop {
    #[primary_span]
    #[suggestion(
        label = "try adding an expression to the `for` loop",
        code = "/* expression */ ",
        applicability = "has-placeholders",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$loop_kind}...else` loops are not supported")]
#[note(
    "consider moving this `else` clause to a separate `if` statement and use a `bool` variable to control if it should run"
)]
pub(crate) struct LoopElseNotSupported {
    #[primary_span]
    pub span: Span,
    pub loop_kind: &'static str,
    #[label("`else` is attached to this loop")]
    pub loop_kw: Span,
}

#[derive(Diagnostic)]
#[diag("expected `,` following `match` arm")]
pub(crate) struct MissingCommaAfterMatchArm {
    #[primary_span]
    #[suggestion(
        label = "missing a comma here to end this `match` arm",
        applicability = "machine-applicable",
        code = ","
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("keyword `catch` cannot follow a `try` block")]
#[help("try using `match` on the result of the `try` block instead")]
pub(crate) struct CatchAfterTry {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`gen` functions are not yet implemented")]
#[help("for now you can use `gen {}` blocks and return `impl Iterator` instead")]
pub(crate) struct GenFn {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`async gen` functions are not supported")]
pub(crate) struct AsyncGenFn {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot use a comma after the base struct")]
#[note("the base struct must always be the last field")]
pub(crate) struct CommaAfterBaseStruct {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "remove this comma",
        style = "short",
        applicability = "machine-applicable",
        code = ""
    )]
    pub comma: Span,
}

#[derive(Diagnostic)]
#[diag("expected `:`, found `=`")]
pub(crate) struct EqFieldInit {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "replace equals symbol with a colon",
        applicability = "machine-applicable",
        code = ":"
    )]
    pub eq: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token: `...`")]
pub(crate) struct DotDotDot {
    #[primary_span]
    #[suggestion(
        label = "use `..` for an exclusive range",
        applicability = "maybe-incorrect",
        code = ".."
    )]
    #[suggestion(
        label = "or `..=` for an inclusive range",
        applicability = "maybe-incorrect",
        code = "..="
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token: `<-`")]
pub(crate) struct LeftArrowOperator {
    #[primary_span]
    #[suggestion(
        label = "if you meant to write a comparison against a negative value, add a space in between `<` and `-`",
        applicability = "maybe-incorrect",
        code = "< -"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected pattern, found `let`")]
pub(crate) struct RemoveLet {
    #[primary_span]
    #[suggestion(
        label = "remove the unnecessary `let` keyword",
        applicability = "machine-applicable",
        code = ""
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `==`")]
pub(crate) struct UseEqInstead {
    #[primary_span]
    #[suggestion(
        label = "try using `=` instead",
        style = "short",
        applicability = "machine-applicable",
        code = "="
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected `{}`, found `;`")]
pub(crate) struct UseEmptyBlockNotSemi {
    #[primary_span]
    #[suggestion(
        label = "try using `{}` instead",
        style = "hidden",
        applicability = "machine-applicable",
        code = "{{}}"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`<` is interpreted as a start of generic arguments for `{$type}`, not a comparison")]
pub(crate) struct ComparisonInterpretedAsGeneric {
    #[primary_span]
    #[label("not interpreted as comparison")]
    pub comparison: Span,
    pub r#type: Path,
    #[label("interpreted as generic arguments")]
    pub args: Span,
    #[subdiagnostic]
    pub suggestion: ComparisonOrShiftInterpretedAsGenericSugg,
    pub action: String,
}

#[derive(Diagnostic)]
#[diag("`<<` is interpreted as a start of generic arguments for `{$type}`, not a shift")]
pub(crate) struct ShiftInterpretedAsGeneric {
    #[primary_span]
    #[label("not interpreted as shift")]
    pub shift: Span,
    pub r#type: Path,
    #[label("interpreted as generic arguments")]
    pub args: Span,
    #[subdiagnostic]
    pub suggestion: ComparisonOrShiftInterpretedAsGenericSugg,
    pub action: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "try {$action} the cast value",
    applicability = "machine-applicable"
)]
pub(crate) struct ComparisonOrShiftInterpretedAsGenericSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("expected expression, found `{$token}`")]
pub(crate) struct FoundExprWouldBeStmt {
    #[primary_span]
    #[label("expected expression")]
    pub span: Span,
    pub token: Token,
    #[subdiagnostic]
    pub suggestion: ExprParenthesesNeeded,
}

#[derive(Diagnostic)]
#[diag("leading `+` is not supported")]
pub(crate) struct LeadingPlusNotSupported {
    #[primary_span]
    #[label("unexpected `+`")]
    pub span: Span,
    #[suggestion(
        label = "try removing the `+`",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub remove_plus: Option<Span>,
    #[subdiagnostic]
    pub add_parentheses: Option<ExprParenthesesNeeded>,
}

#[derive(Diagnostic)]
#[diag("invalid `struct` delimiters or `fn` call arguments")]
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
#[multipart_suggestion(
    label = "if `{$type}` is a struct, use braces as delimiters",
    applicability = "maybe-incorrect"
)]
pub(crate) struct BracesForStructLiteral {
    #[suggestion_part(code = " {{ ")]
    pub first: Span,
    #[suggestion_part(code = " }}")]
    pub second: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "if `{$type}` is a function, use the arguments directly",
    applicability = "maybe-incorrect"
)]
pub(crate) struct NoFieldsForFnCall {
    #[suggestion_part(code = "")]
    pub fields: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(
    "parentheses are required around this expression to avoid confusion with a labeled break expression"
)]
pub(crate) struct LabeledLoopInBreak {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: WrapExpressionInParentheses,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "wrap the expression in parentheses",
    applicability = "machine-applicable"
)]
pub(crate) struct WrapExpressionInParentheses {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("this is a block expression, not an array")]
pub(crate) struct ArrayBracketsInsteadOfSpaces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: ArrayBracketsInsteadOfSpacesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "to make an array, use square brackets instead of curly braces",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ArrayBracketsInsteadOfSpacesSugg {
    #[suggestion_part(code = "[")]
    pub left: Span,
    #[suggestion_part(code = "]")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("`match` arm body without braces")]
pub(crate) struct MatchArmBodyWithoutBraces {
    #[primary_span]
    #[label(
        "{$num_statements ->
[one] this statement is not surrounded by a body
*[other] these statements are not surrounded by a body
}"
    )]
    pub statements: Span,
    #[label("while parsing the `match` arm starting here")]
    pub arrow: Span,
    pub num_statements: usize,
    #[subdiagnostic]
    pub sub: MatchArmBodyWithoutBracesSugg,
}

#[derive(Diagnostic)]
#[diag("unexpected `=` after inclusive range")]
#[note("inclusive ranges end with a single equals sign (`..=`)")]
pub(crate) struct InclusiveRangeExtraEquals {
    #[primary_span]
    #[suggestion(
        label = "use `..=` instead",
        style = "short",
        code = "..=",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `>` after inclusive range")]
pub(crate) struct InclusiveRangeMatchArrow {
    #[primary_span]
    pub arrow: Span,
    #[label("this is parsed as an inclusive range `..=`")]
    pub span: Span,
    #[suggestion(
        label = "add a space between the pattern and `=>`",
        style = "verbose",
        code = " ",
        applicability = "machine-applicable"
    )]
    pub after_pat: Span,
}

#[derive(Diagnostic)]
#[diag(label = "inclusive range with no end", code = "E0586")]
#[note("inclusive ranges must be bounded at the end (`..=b` or `a..=b`)")]
pub(crate) struct InclusiveRangeNoEnd {
    #[primary_span]
    #[suggestion(
        label = "use `..` instead",
        code = "..",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum MatchArmBodyWithoutBracesSugg {
    #[multipart_suggestion(
        label = "surround the {$num_statements ->
[one] statement
*[other] statements
} with a body",
        applicability = "machine-applicable"
    )]
    AddBraces {
        #[suggestion_part(code = "{{ ")]
        left: Span,
        #[suggestion_part(code = " }}")]
        right: Span,
    },
    #[suggestion(
        label = "replace `;` with `,` to end a `match` arm expression",
        code = ",",
        applicability = "machine-applicable"
    )]
    UseComma {
        #[primary_span]
        semicolon: Span,
    },
}

#[derive(Diagnostic)]
#[diag("struct literals are not allowed here")]
pub(crate) struct StructLiteralNotAllowedHere {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: StructLiteralNotAllowedHereSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "surround the struct literal with parentheses",
    applicability = "machine-applicable"
)]
pub(crate) struct StructLiteralNotAllowedHereSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("invalid interpolated expression")]
pub(crate) struct InvalidInterpolatedExpression {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("suffixes on a tuple index are invalid")]
pub(crate) struct InvalidLiteralSuffixOnTupleIndex {
    #[primary_span]
    #[label("invalid suffix `{$suffix}`")]
    pub span: Span,
    pub suffix: Symbol,
    #[help(
        "`{$suffix}` is *temporarily* accepted on tuple index fields as it was incorrectly accepted on stable for a few releases"
    )]
    #[help(
        "on proc macros, you'll want to use `syn::Index::from` or `proc_macro::Literal::*_unsuffixed` for code that will desugar to tuple field access"
    )]
    #[help(
        "see issue #60210 <https://github.com/rust-lang/rust/issues/60210> for more information"
    )]
    pub exception: Option<()>,
}

#[derive(Diagnostic)]
#[diag("non-string ABI literal")]
pub(crate) struct NonStringAbiLiteral {
    #[primary_span]
    #[suggestion(
        label = "specify the ABI with a string literal",
        code = "\"C\"",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("mismatched closing delimiter: `{$delimiter}`")]
pub(crate) struct MismatchedClosingDelimiter {
    #[primary_span]
    pub spans: Vec<Span>,
    pub delimiter: String,
    #[label("mismatched closing delimiter")]
    pub unmatched: Span,
    #[label("closing delimiter possibly meant for this")]
    pub opening_candidate: Option<Span>,
    #[label("unclosed delimiter")]
    pub unclosed: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(label = "incorrect visibility restriction", code = "E0704")]
#[help(
    "some possible visibility restrictions are:
`pub(crate)`: visible only on the current crate
`pub(super)`: visible only in the current module's parent
`pub(in path::to::module)`: visible only on the specified path"
)]
pub(crate) struct IncorrectVisibilityRestriction {
    #[primary_span]
    #[suggestion(
        label = "make this visible only to module `{$inner_str}` with `in`",
        code = "in {inner_str}",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub inner_str: String,
}

#[derive(Diagnostic)]
#[diag("<assignment> ... else { ... } is not allowed")]
pub(crate) struct AssignmentElseNotAllowed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected statement after outer attribute")]
pub(crate) struct ExpectedStatementAfterOuterAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(label = "found a documentation comment that doesn't document anything", code = "E0585")]
#[help("doc comments must come before what they document, if a comment was intended use `//`")]
pub(crate) struct DocCommentDoesNotDocumentAnything {
    #[primary_span]
    pub span: Span,
    #[suggestion(label = "missing comma here", code = ",", applicability = "machine-applicable")]
    pub missing_comma: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("`const` and `let` are mutually exclusive")]
pub(crate) struct ConstLetMutuallyExclusive {
    #[primary_span]
    #[suggestion(label = "remove `let`", code = "const", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("a `{$operator}` expression cannot be directly assigned in `let...else`")]
pub(crate) struct InvalidExpressionInLetElse {
    #[primary_span]
    pub span: Span,
    pub operator: &'static str,
    #[subdiagnostic]
    pub sugg: WrapExpressionInParentheses,
}

#[derive(Diagnostic)]
#[diag("right curly brace `}` before `else` in a `let...else` statement not allowed")]
pub(crate) struct InvalidCurlyInLetElse {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WrapExpressionInParentheses,
}

#[derive(Diagnostic)]
#[diag("can't reassign to an uninitialized variable")]
#[help("if you meant to overwrite, remove the `let` binding")]
pub(crate) struct CompoundAssignmentExpressionInLet {
    #[primary_span]
    #[suggestion(
        label = "initialize the variable",
        style = "short",
        code = "=",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("suffixed literals are not allowed in attributes")]
#[help(
    "instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), use an unsuffixed version (`1`, `1.0`, etc.)"
)]
pub(crate) struct SuffixedLiteralInAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected unsuffixed literal or identifier, found `{$token}`")]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub token: Token,
}

#[derive(Subdiagnostic)]
#[suggestion(
    label = "escape `{$ident_name}` to use it as an identifier",
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
#[suggestion(label = "remove this comma", applicability = "machine-applicable", code = "")]
pub(crate) struct SuggRemoveComma {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    label = "you might have meant to introduce a new binding",
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
    #[label("expected identifier, found reserved identifier")]
    ReservedIdentifier(#[primary_span] Span),
    #[label("expected identifier, found keyword")]
    Keyword(#[primary_span] Span),
    #[label("expected identifier, found reserved keyword")]
    ReservedKeyword(#[primary_span] Span),
    #[label("expected identifier, found doc comment")]
    DocComment(#[primary_span] Span),
    #[label("expected identifier")]
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

impl<'a> IntoDiagnostic<'a> for ExpectedIdentifier {
    #[track_caller]
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a> {
        let token_descr = TokenDescription::from_token(&self.token);

        let mut diag = DiagnosticBuilder::new(
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
        diag.set_span(self.span);
        diag.set_arg("token", self.token);

        if let Some(sugg) = self.suggest_raw {
            sugg.add_to_diagnostic(&mut diag);
        }

        ExpectedIdentifierFound::new(token_descr, self.span).add_to_diagnostic(&mut diag);

        if let Some(sugg) = self.suggest_remove_comma {
            sugg.add_to_diagnostic(&mut diag);
        }

        if let Some(help) = self.help_cannot_start_number {
            help.add_to_diagnostic(&mut diag);
        }

        diag
    }
}

#[derive(Subdiagnostic)]
#[help("identifiers cannot start with a number")]
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

impl<'a> IntoDiagnostic<'a> for ExpectedSemi {
    #[track_caller]
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a> {
        let token_descr = TokenDescription::from_token(&self.token);

        let mut diag = DiagnosticBuilder::new(
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
        diag.set_span(self.span);
        diag.set_arg("token", self.token);

        if let Some(unexpected_token_label) = self.unexpected_token_label {
            diag.span_label(unexpected_token_label, fluent::parse_label_unexpected_token);
        }

        self.sugg.add_to_diagnostic(&mut diag);

        diag
    }
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedSemiSugg {
    #[suggestion(label = "change this to `;`", code = ";", applicability = "machine-applicable")]
    ChangeToSemi(#[primary_span] Span),
    #[suggestion(
        label = "add `;` here",
        style = "short",
        code = ";",
        applicability = "machine-applicable"
    )]
    AddSemi(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("struct literal body without path")]
pub(crate) struct StructLiteralBodyWithoutPath {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: StructLiteralBodyWithoutPathSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "you might have forgotten to add the struct literal inside the block",
    applicability = "has-placeholders"
)]
pub(crate) struct StructLiteralBodyWithoutPathSugg {
    #[suggestion_part(code = "{{ SomeStruct ")]
    pub before: Span,
    #[suggestion_part(code = " }}")]
    pub after: Span,
}

#[derive(Diagnostic)]
#[diag("invalid struct literal")]
pub(crate) struct StructLiteralNeedingParens {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: StructLiteralNeedingParensSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "you might need to surround the struct literal with parentheses",
    applicability = "machine-applicable"
)]
pub(crate) struct StructLiteralNeedingParensSugg {
    #[suggestion_part(code = "(")]
    pub before: Span,
    #[suggestion_part(code = ")")]
    pub after: Span,
}

#[derive(Diagnostic)]
#[diag(
    "{$num_extra_brackets ->
[one] unmatched angle bracket
*[other] unmatched angle brackets
}"
)]
pub(crate) struct UnmatchedAngleBrackets {
    #[primary_span]
    #[suggestion(
        label = "{$num_extra_brackets ->
[one] remove extra angle bracket
*[other] remove extra angle brackets
}",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub num_extra_brackets: usize,
}

#[derive(Diagnostic)]
#[diag("generic parameters without surrounding angle brackets")]
pub(crate) struct GenericParamsWithoutAngleBrackets {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: GenericParamsWithoutAngleBracketsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "surround the type parameters with angle brackets",
    applicability = "machine-applicable"
)]
pub(crate) struct GenericParamsWithoutAngleBracketsSugg {
    #[suggestion_part(code = "<")]
    pub left: Span,
    #[suggestion_part(code = ">")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("comparison operators cannot be chained")]
pub(crate) struct ComparisonOperatorsCannotBeChained {
    #[primary_span]
    pub span: Vec<Span>,
    #[suggestion(
        label = "use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments",
        style = "verbose",
        code = "::",
        applicability = "maybe-incorrect"
    )]
    pub suggest_turbofish: Option<Span>,
    #[help("use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments")]
    #[help("or use `(...)` if you meant to specify fn arguments")]
    pub help_turbofish: Option<()>,
    #[subdiagnostic]
    pub chaining_sugg: Option<ComparisonOperatorsCannotBeChainedSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ComparisonOperatorsCannotBeChainedSugg {
    #[suggestion(
        label = "split the comparison into two",
        style = "verbose",
        code = " && {middle_term}",
        applicability = "maybe-incorrect"
    )]
    SplitComparison {
        #[primary_span]
        span: Span,
        middle_term: String,
    },
    #[multipart_suggestion(
        label = "parenthesize the comparison",
        applicability = "maybe-incorrect"
    )]
    Parenthesize {
        #[suggestion_part(code = "(")]
        left: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
}

#[derive(Diagnostic)]
#[diag("invalid `?` in type")]
pub(crate) struct QuestionMarkInType {
    #[primary_span]
    #[label("`?` is only allowed on expressions, not types")]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: QuestionMarkInTypeSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "if you meant to express that the type might not contain a value, use the `Option` wrapper type",
    applicability = "machine-applicable"
)]
pub(crate) struct QuestionMarkInTypeSugg {
    #[suggestion_part(code = "Option<")]
    pub left: Span,
    #[suggestion_part(code = ">")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected parentheses surrounding `for` loop head")]
pub(crate) struct ParenthesesInForHead {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: ParenthesesInForHeadSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "remove parentheses in `for` loop",
    applicability = "machine-applicable"
)]
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
#[diag("documentation comments cannot be applied to a function parameter's type")]
pub(crate) struct DocCommentOnParamType {
    #[primary_span]
    #[label("doc comments are not allowed here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attributes cannot be applied to a function parameter's type")]
pub(crate) struct AttributeOnParamType {
    #[primary_span]
    #[label("attributes are not allowed here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(label = "patterns aren't allowed in methods without bodies", code = "E0642")]
pub(crate) struct PatternMethodParamWithoutBody {
    #[primary_span]
    #[suggestion(
        label = "give this argument a name or use an underscore to ignore it",
        code = "_",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `self` parameter in function")]
pub(crate) struct SelfParamNotFirst {
    #[primary_span]
    #[label("must be the first parameter of an associated function")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expressions must be enclosed in braces to be used as const generic arguments")]
pub(crate) struct ConstGenericWithoutBraces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ConstGenericWithoutBracesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "enclose the `const` expression in braces",
    applicability = "machine-applicable"
)]
pub(crate) struct ConstGenericWithoutBracesSugg {
    #[suggestion_part(code = "{{ ")]
    pub left: Span,
    #[suggestion_part(code = " }}")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `const` parameter declaration")]
pub(crate) struct UnexpectedConstParamDeclaration {
    #[primary_span]
    #[label("expected a `const` expression, not a parameter declaration")]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<UnexpectedConstParamDeclarationSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnexpectedConstParamDeclarationSugg {
    #[multipart_suggestion(
        label = "`const` parameters must be declared for the `impl`",
        applicability = "machine-applicable"
    )]
    AddParam {
        #[suggestion_part(code = "<{snippet}>")]
        impl_generics: Span,
        #[suggestion_part(code = "{ident}")]
        incorrect_decl: Span,
        snippet: String,
        ident: String,
    },
    #[multipart_suggestion(
        label = "`const` parameters must be declared for the `impl`",
        applicability = "machine-applicable"
    )]
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
#[diag("expected lifetime, type, or constant, found keyword `const`")]
pub(crate) struct UnexpectedConstInGenericParam {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "the `const` keyword is only needed in the definition of the type",
        style = "verbose",
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub to_remove: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("the order of `move` and `async` is incorrect")]
pub(crate) struct AsyncMoveOrderIncorrect {
    #[primary_span]
    #[suggestion(
        label = "try switching the order",
        style = "verbose",
        code = "async move",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected `:` followed by trait or lifetime")]
pub(crate) struct DoubleColonInBound {
    #[primary_span]
    pub span: Span,
    #[suggestion(label = "use single colon", code = ": ", applicability = "machine-applicable")]
    pub between: Span,
}

#[derive(Diagnostic)]
#[diag("function pointer types may not have generic parameters")]
pub(crate) struct FnPtrWithGenerics {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<FnPtrWithGenericsSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "consider moving the lifetime {$arity ->
[one] parameter
*[other] parameters
} to {$for_param_list_exists ->
[true] the
*[false] a
} `for` parameter list",
    applicability = "maybe-incorrect"
)]
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
    pub machine_applicable: bool,
}

impl AddToDiagnostic for FnTraitMissingParen {
    fn add_to_diagnostic_with<F>(self, diag: &mut rustc_errors::Diagnostic, _: F)
    where
        F: Fn(&mut rustc_errors::Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        diag.span_label(self.span, crate::fluent_generated::parse_fn_trait_missing_paren);
        let applicability = if self.machine_applicable {
            Applicability::MachineApplicable
        } else {
            Applicability::MaybeIncorrect
        };
        diag.span_suggestion_short(
            self.span.shrink_to_hi(),
            crate::fluent_generated::parse_add_paren,
            "()",
            applicability,
        );
    }
}

#[derive(Diagnostic)]
#[diag("unexpected `if` in the condition expression")]
pub(crate) struct UnexpectedIfWithIf(
    #[primary_span]
    #[suggestion(
        label = "remove the `if`",
        applicability = "machine-applicable",
        code = " ",
        style = "verbose"
    )]
    pub Span,
);

#[derive(Diagnostic)]
#[diag("you might have meant to write `impl` instead of `fn`")]
pub(crate) struct FnTypoWithImpl {
    #[primary_span]
    #[suggestion(
        label = "replace `fn` with `impl` here",
        applicability = "maybe-incorrect",
        code = "impl",
        style = "verbose"
    )]
    pub fn_span: Span,
}

#[derive(Diagnostic)]
#[diag("expected identifier, found keyword `fn`")]
pub(crate) struct ExpectedFnPathFoundFnKeyword {
    #[primary_span]
    #[suggestion(
        label = "use `Fn` to refer to the trait",
        applicability = "machine-applicable",
        code = "Fn",
        style = "verbose"
    )]
    pub fn_token_span: Span,
}

#[derive(Diagnostic)]
#[diag("path separator must be a double colon")]
pub(crate) struct PathSingleColon {
    #[primary_span]
    #[suggestion(
        label = "use a double colon instead",
        applicability = "machine-applicable",
        code = "::"
    )]
    pub span: Span,

    #[note(
        "if you meant to annotate an expression with a type, the type ascription syntax has been removed, see issue #101728 <https://github.com/rust-lang/rust/issues/101728>"
    )]
    pub type_ascription: Option<()>,
}

#[derive(Diagnostic)]
#[diag("statements are terminated with a semicolon")]
pub(crate) struct ColonAsSemi {
    #[primary_span]
    #[suggestion(
        label = "use a semicolon instead",
        applicability = "machine-applicable",
        code = ";"
    )]
    pub span: Span,

    #[note(
        "if you meant to annotate an expression with a type, the type ascription syntax has been removed, see issue #101728 <https://github.com/rust-lang/rust/issues/101728>"
    )]
    pub type_ascription: Option<()>,
}

#[derive(Diagnostic)]
#[diag("where clauses are not allowed before tuple struct bodies")]
pub(crate) struct WhereClauseBeforeTupleStructBody {
    #[primary_span]
    #[label("unexpected where clause")]
    pub span: Span,
    #[label("while parsing this tuple struct")]
    pub name: Span,
    #[label("the struct body")]
    pub body: Span,
    #[subdiagnostic]
    pub sugg: Option<WhereClauseBeforeTupleStructBodySugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "move the body before the where clause",
    applicability = "machine-applicable"
)]
pub(crate) struct WhereClauseBeforeTupleStructBodySugg {
    #[suggestion_part(code = "{snippet}")]
    pub left: Span,
    pub snippet: String,
    #[suggestion_part(code = "")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag(label = "`async fn` is not permitted in Rust 2015", code = "E0670")]
pub(crate) struct AsyncFnIn2015 {
    #[primary_span]
    #[label("to use `async fn`, switch to Rust 2018 or later")]
    pub span: Span,
    #[subdiagnostic]
    pub help: HelpUseLatestEdition,
}

#[derive(Subdiagnostic)]
#[label("`async` blocks are only allowed in Rust 2018 or later")]
pub(crate) struct AsyncBlockIn2015 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`async move` blocks are only allowed in Rust 2018 or later")]
pub(crate) struct AsyncMoveBlockIn2015 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot pass `self` by raw pointer")]
pub(crate) struct SelfArgumentPointer {
    #[primary_span]
    #[label("cannot pass `self` by raw pointer")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token: `{$actual}`")]
pub struct UnexpectedTokenAfterDot<'a> {
    #[primary_span]
    pub span: Span,
    pub actual: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag("visibility `{$vis}` is not followed by an item")]
#[help("you likely meant to define an item, e.g., `{$vis} fn foo() {\"{}\"}`")]
pub(crate) struct VisibilityNotFollowedByItem {
    #[primary_span]
    #[label("the visibility")]
    pub span: Span,
    pub vis: Visibility,
}

#[derive(Diagnostic)]
#[diag("`default` is not followed by an item")]
#[note("only `fn`, `const`, `type`, or `impl` items may be prefixed by `default`")]
pub(crate) struct DefaultNotFollowedByItem {
    #[primary_span]
    #[label("the `default` qualifier")]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum MissingKeywordForItemDefinition {
    #[diag("missing `struct` for struct definition")]
    Struct {
        #[primary_span]
        #[suggestion(
            label = "add `struct` here to parse `{$ident}` as a public struct",
            style = "short",
            applicability = "maybe-incorrect",
            code = " struct "
        )]
        span: Span,
        ident: Ident,
    },
    #[diag("missing `fn` for function definition")]
    Function {
        #[primary_span]
        #[suggestion(
            label = "add `fn` here to parse `{$ident}` as a public function",
            style = "short",
            applicability = "maybe-incorrect",
            code = " fn "
        )]
        span: Span,
        ident: Ident,
    },
    #[diag("missing `fn` for method definition")]
    Method {
        #[primary_span]
        #[suggestion(
            label = "add `fn` here to parse `{$ident}` as a public method",
            style = "short",
            applicability = "maybe-incorrect",
            code = " fn "
        )]
        span: Span,
        ident: Ident,
    },
    #[diag("missing `fn` or `struct` for function or struct definition")]
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
        label = "if you meant to call a macro, try",
        applicability = "maybe-incorrect",
        code = "{snippet}!"
    )]
    SuggestMacro {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[help(
        "if you meant to call a macro, remove the `pub` and add a trailing `!` after the identifier"
    )]
    HelpMacro,
}

#[derive(Diagnostic)]
#[diag("missing parameters for function definition")]
pub(crate) struct MissingFnParams {
    #[primary_span]
    #[suggestion(
        label = "add a parameter list",
        code = "()",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("missing trait in a trait impl")]
pub(crate) struct MissingTraitInTraitImpl {
    #[primary_span]
    #[suggestion(label = "add a trait here", code = " Trait ", applicability = "has-placeholders")]
    pub span: Span,
    #[suggestion(
        label = "for an inherent impl, drop this `for`",
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub for_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing `for` in a trait impl")]
pub(crate) struct MissingForInTraitImpl {
    #[primary_span]
    #[suggestion(
        label = "add `for` here",
        style = "short",
        code = " for ",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a trait, found type")]
pub(crate) struct ExpectedTraitInTraitImplFoundType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `impl` keyword")]
pub(crate) struct ExtraImplKeywordInTraitImpl {
    #[primary_span]
    #[suggestion(label = "remove the extra `impl`", code = "", applicability = "maybe-incorrect")]
    pub extra_impl_kw: Span,
    #[note("this is parsed as an `impl Trait` type, but a trait is expected at this position")]
    pub impl_trait_span: Span,
}

#[derive(Diagnostic)]
#[diag("bounds are not allowed on trait aliases")]
pub(crate) struct BoundsNotAllowedOnTraitAliases {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("trait aliases cannot be `auto`")]
pub(crate) struct TraitAliasCannotBeAuto {
    #[primary_span]
    #[label("trait aliases cannot be `auto`")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("trait aliases cannot be `unsafe`")]
pub(crate) struct TraitAliasCannotBeUnsafe {
    #[primary_span]
    #[label("trait aliases cannot be `unsafe`")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("associated `static` items are not allowed")]
pub(crate) struct AssociatedStaticItemNotAllowed {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("crate name using dashes are not valid in `extern crate` statements")]
pub(crate) struct ExternCrateNameWithDashes {
    #[primary_span]
    #[label("dash-separated idents are not valid")]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: ExternCrateNameWithDashesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "if the original crate name uses dashes you need to use underscores in the code",
    applicability = "machine-applicable"
)]
pub(crate) struct ExternCrateNameWithDashesSugg {
    #[suggestion_part(code = "_")]
    pub dashes: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("extern items cannot be `const`")]
#[note("for more information, visit https://doc.rust-lang.org/std/keyword.extern.html")]
pub(crate) struct ExternItemCannotBeConst {
    #[primary_span]
    pub ident_span: Span,
    #[suggestion(
        label = "try using a static value",
        code = "static ",
        applicability = "machine-applicable"
    )]
    pub const_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("const globals cannot be mutable")]
pub(crate) struct ConstGlobalCannotBeMutable {
    #[primary_span]
    #[label("cannot be mutable")]
    pub ident_span: Span,
    #[suggestion(
        label = "you might want to declare a static instead",
        code = "static",
        applicability = "maybe-incorrect"
    )]
    pub const_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing type for `{$kind}` item")]
pub(crate) struct MissingConstType {
    #[primary_span]
    #[suggestion(
        label = "provide a type for the item",
        code = "{colon} <type>",
        applicability = "has-placeholders"
    )]
    pub span: Span,

    pub kind: &'static str,
    pub colon: &'static str,
}

#[derive(Diagnostic)]
#[diag("`enum` and `struct` are mutually exclusive")]
pub(crate) struct EnumStructMutuallyExclusive {
    #[primary_span]
    #[suggestion(
        label = "replace `enum struct` with",
        code = "enum",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum UnexpectedTokenAfterStructName {
    #[diag(
        label = r#"expected `where`, `{"{"}`, `(`, or `;` after struct name, found reserved identifier `{$token}`"#
    )]
    ReservedIdentifier {
        #[primary_span]
        #[label(r#"expected `where`, `{"{"}`, `(`, or `;` after struct name"#)]
        span: Span,
        token: Token,
    },
    #[diag(
        label = r#"expected `where`, `{"{"}`, `(`, or `;` after struct name, found keyword `{$token}`"#
    )]
    Keyword {
        #[primary_span]
        #[label(r#"expected `where`, `{"{"}`, `(`, or `;` after struct name"#)]
        span: Span,
        token: Token,
    },
    #[diag(
        label = r#"expected `where`, `{"{"}`, `(`, or `;` after struct name, found reserved keyword `{$token}`"#
    )]
    ReservedKeyword {
        #[primary_span]
        #[label(r#"expected `where`, `{"{"}`, `(`, or `;` after struct name"#)]
        span: Span,
        token: Token,
    },
    #[diag(
        label = r#"expected `where`, `{"{"}`, `(`, or `;` after struct name, found doc comment `{$token}`"#
    )]
    DocComment {
        #[primary_span]
        #[label(r#"expected `where`, `{"{"}`, `(`, or `;` after struct name"#)]
        span: Span,
        token: Token,
    },
    #[diag(
        label = r#"expected `where`, `{"{"}`, `(`, or `;` after struct name, found `{$token}`"#
    )]
    Other {
        #[primary_span]
        #[label(r#"expected `where`, `{"{"}`, `(`, or `;` after struct name"#)]
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
#[diag("unexpected keyword `Self` in generic parameters")]
#[note("you cannot use `Self` as a generic parameter because it is reserved for associated items")]
pub(crate) struct UnexpectedSelfInGenericParameters {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected default lifetime parameter")]
pub(crate) struct UnexpectedDefaultValueForLifetimeInGenericParameters {
    #[primary_span]
    #[label("lifetime parameters cannot have default values")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot define duplicate `where` clauses on an item")]
pub(crate) struct MultipleWhereClauses {
    #[primary_span]
    pub span: Span,
    #[label("previous `where` clause starts here")]
    pub previous: Span,
    #[suggestion(
        label = "consider joining the two `where` clauses into one",
        style = "verbose",
        code = ",",
        applicability = "maybe-incorrect"
    )]
    pub between: Span,
}

#[derive(Diagnostic)]
pub(crate) enum UnexpectedNonterminal {
    #[diag("expected an item keyword")]
    Item(#[primary_span] Span),
    #[diag("expected a statement")]
    Statement(#[primary_span] Span),
    #[diag("expected ident, found `{$token}`")]
    Ident {
        #[primary_span]
        span: Span,
        token: Token,
    },
    #[diag("expected a lifetime, found `{$token}`")]
    Lifetime {
        #[primary_span]
        span: Span,
        token: Token,
    },
}

#[derive(Diagnostic)]
pub(crate) enum TopLevelOrPatternNotAllowed {
    #[diag("top-level or-patterns are not allowed in `let` bindings")]
    LetBinding {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        sub: Option<TopLevelOrPatternNotAllowedSugg>,
    },
    #[diag("top-level or-patterns are not allowed in function parameters")]
    FunctionParameter {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        sub: Option<TopLevelOrPatternNotAllowedSugg>,
    },
}

#[derive(Diagnostic)]
#[diag("`{$ident}` cannot be a raw identifier")]
pub struct CannotBeRawIdent {
    #[primary_span]
    pub span: Span,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "bare CR not allowed in {$block ->
[true] block doc-comment
*[false] doc-comment
}"
)]
pub struct CrDocComment {
    #[primary_span]
    pub span: Span,
    pub block: bool,
}

#[derive(Diagnostic)]
#[diag(label = "no valid digits found for number", code = "E0768")]
pub struct NoDigitsLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid digit for a base {$base} literal")]
pub struct InvalidDigitLiteral {
    #[primary_span]
    pub span: Span,
    pub base: u32,
}

#[derive(Diagnostic)]
#[diag("expected at least one digit in exponent")]
pub struct EmptyExponentFloat {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$base} float literal is not supported")]
pub struct FloatLiteralUnsupportedBase {
    #[primary_span]
    pub span: Span,
    pub base: &'static str,
}

#[derive(Diagnostic)]
#[diag("prefix `{$prefix}` is unknown")]
#[note("prefixed identifiers and literals are reserved since Rust 2021")]
pub struct UnknownPrefix<'a> {
    #[primary_span]
    #[label("unknown prefix")]
    pub span: Span,
    pub prefix: &'a str,
    #[subdiagnostic]
    pub sugg: Option<UnknownPrefixSugg>,
}

#[derive(Subdiagnostic)]
#[note("macros cannot expand to {$adt_ty} fields")]
pub struct MacroExpandsToAdtField<'a> {
    pub adt_ty: &'a str,
}

#[derive(Subdiagnostic)]
pub enum UnknownPrefixSugg {
    #[suggestion(
        label = "use `br` for a raw byte string",
        code = "br",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    UseBr(#[primary_span] Span),
    #[suggestion(
        label = "consider inserting whitespace here",
        code = " ",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Whitespace(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag(
    "too many `#` symbols: raw strings may be delimited by up to 255 `#` symbols, but found {$num}"
)]
pub struct TooManyHashes {
    #[primary_span]
    pub span: Span,
    pub num: u32,
}

#[derive(Diagnostic)]
#[diag("unknown start of token: {$escaped}")]
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
        label = "Unicode characters '“' (Left Double Quotation Mark) and '”' (Right Double Quotation Mark) look like '{$ascii_str}' ({$ascii_name}), but are not",
        code = "{suggestion}",
        applicability = "maybe-incorrect"
    )]
    DirectedQuotes {
        #[primary_span]
        span: Span,
        suggestion: String,
        ascii_str: &'static str,
        ascii_name: &'static str,
    },
    #[suggestion(
        label = "Unicode character '{$ch}' ({$u_name}) looks like '{$ascii_str}' ({$ascii_name}), but it is not",
        code = "{suggestion}",
        applicability = "maybe-incorrect"
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
#[note(
    "character appears {$repeats ->
[one] once more
*[other] {$repeats} more times
}"
)]
pub struct UnknownTokenRepeat {
    pub repeats: usize,
}

#[derive(Subdiagnostic)]
#[help(
    "source files must contain UTF-8 encoded text, unexpected null bytes might occur when a different encoding is used"
)]
pub struct UnknownTokenNull;

#[derive(Diagnostic)]
pub enum UnescapeError {
    #[diag("invalid unicode character escape")]
    #[help(
        "unicode escape must {$surrogate ->
[true] not be a surrogate
*[false] be at most 10FFFF
}"
    )]
    InvalidUnicodeEscape {
        #[primary_span]
        #[label("invalid escape")]
        span: Span,
        surrogate: bool,
    },
    #[diag(
        "{$byte ->
[true] byte
*[false] character
} constant must be escaped: `{$escaped_msg}`"
    )]
    EscapeOnlyChar {
        #[primary_span]
        span: Span,
        #[suggestion(
            label = "escape the character",
            applicability = "machine-applicable",
            code = "{escaped_sugg}"
        )]
        char_span: Span,
        escaped_sugg: String,
        escaped_msg: String,
        byte: bool,
    },
    #[diag(
        "{$double_quotes ->
[true] bare CR not allowed in string, use `\r` instead
*[false] character constant must be escaped: `\r`
}"
    )]
    BareCr {
        #[primary_span]
        #[suggestion(
            label = "escape the character",
            applicability = "machine-applicable",
            code = "\\r"
        )]
        span: Span,
        double_quotes: bool,
    },
    #[diag("bare CR not allowed in raw string")]
    BareCrRawString(#[primary_span] Span),
    #[diag("numeric character escape is too short")]
    TooShortHexEscape(#[primary_span] Span),
    #[diag(
        "invalid character in {$is_hex ->
[true] numeric character
*[false] unicode
} escape: `{$ch}`"
    )]
    InvalidCharInEscape {
        #[primary_span]
        #[label(
            "invalid character in {$is_hex ->
[true] numeric character
*[false] unicode
} escape"
        )]
        span: Span,
        is_hex: bool,
        ch: String,
    },
    #[diag("out of range hex escape")]
    OutOfRangeHexEscape(
        #[primary_span]
        #[label(r#"must be a character in the range [\x00-\x7f]"#)]
        Span,
    ),
    #[diag("invalid start of unicode escape: `_`")]
    LeadingUnderscoreUnicodeEscape {
        #[primary_span]
        #[label("invalid start of unicode escape")]
        span: Span,
        ch: String,
    },
    #[diag("overlong unicode escape")]
    OverlongUnicodeEscape(
        #[primary_span]
        #[label("must have at most 6 hex digits")]
        Span,
    ),
    #[diag("unterminated unicode escape")]
    UnclosedUnicodeEscape(
        #[primary_span]
        #[label("missing a closing `}`")]
        Span,
        #[suggestion(
            label = "terminate the unicode escape",
            code = "}}",
            applicability = "maybe-incorrect",
            style = "verbose"
        )]
        Span,
    ),
    #[diag("incorrect unicode escape sequence")]
    NoBraceInUnicodeEscape {
        #[primary_span]
        span: Span,
        #[label("incorrect unicode escape sequence")]
        label: Option<Span>,
        #[subdiagnostic]
        sub: NoBraceUnicodeSub,
    },
    #[diag("unicode escape in byte string")]
    #[help("unicode escape sequences cannot be used as a byte or in a byte string")]
    UnicodeEscapeInByte(
        #[primary_span]
        #[label("unicode escape in byte string")]
        Span,
    ),
    #[diag("empty unicode escape")]
    EmptyUnicodeEscape(
        #[primary_span]
        #[label("this escape must have at least 1 hex digit")]
        Span,
    ),
    #[diag("empty character literal")]
    ZeroChars(
        #[primary_span]
        #[label("empty character literal")]
        Span,
    ),
    #[diag("invalid trailing slash in literal")]
    LoneSlash(
        #[primary_span]
        #[label("invalid trailing slash in literal")]
        Span,
    ),
    #[diag("whitespace symbol '{$ch}' is not skipped")]
    UnskippedWhitespace {
        #[primary_span]
        span: Span,
        #[label("whitespace symbol '{$ch}' is not skipped")]
        char_span: Span,
        ch: String,
    },
    #[diag("multiple lines skipped by escaped newline")]
    MultipleSkippedLinesWarning(
        #[primary_span]
        #[label("skipping everything up to and including this point")]
        Span,
    ),
    #[diag("character literal may only contain one codepoint")]
    MoreThanOneChar {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        note: Option<MoreThanOneCharNote>,
        #[subdiagnostic]
        suggestion: MoreThanOneCharSugg,
    },
}

#[derive(Subdiagnostic)]
pub enum MoreThanOneCharSugg {
    #[suggestion(
        label = "consider using the normalized form `{$ch}` of this character",
        code = "{normalized}",
        applicability = "machine-applicable"
    )]
    NormalizedForm {
        #[primary_span]
        span: Span,
        ch: String,
        normalized: String,
    },
    #[suggestion(
        label = "consider removing the non-printing characters",
        code = "{ch}",
        applicability = "maybe-incorrect"
    )]
    RemoveNonPrinting {
        #[primary_span]
        span: Span,
        ch: String,
    },
    #[suggestion(
        label = "if you meant to write a {$is_byte ->
[true] byte string
*[false] `str`
} literal, use double quotes",
        code = "{sugg}",
        applicability = "machine-applicable"
    )]
    Quotes {
        #[primary_span]
        span: Span,
        is_byte: bool,
        sugg: String,
    },
}

#[derive(Subdiagnostic)]
pub enum MoreThanOneCharNote {
    #[note(
        "this `{$chr}` is followed by the combining {$len ->
[one] mark
*[other] marks
} `{$escaped_marks}`"
    )]
    AllCombining {
        #[primary_span]
        span: Span,
        chr: String,
        len: usize,
        escaped_marks: String,
    },
    #[note("there are non-printing characters, the full sequence is `{$escaped}`")]
    NonPrinting {
        #[primary_span]
        span: Span,
        escaped: String,
    },
}

#[derive(Subdiagnostic)]
pub enum NoBraceUnicodeSub {
    #[suggestion(
        label = "format of unicode escape sequences uses braces",
        code = "{suggestion}",
        applicability = "maybe-incorrect"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion: String,
    },
    #[help(r#"format of unicode escape sequences is `\u{...}`"#)]
    Help,
}

#[derive(Subdiagnostic)]
pub(crate) enum TopLevelOrPatternNotAllowedSugg {
    #[suggestion(label = "remove the `|`", code = "{pat}", applicability = "machine-applicable")]
    RemoveLeadingVert {
        #[primary_span]
        span: Span,
        pat: String,
    },
    #[suggestion(
        label = "wrap the pattern in parentheses",
        code = "({pat})",
        applicability = "machine-applicable"
    )]
    WrapInParens {
        #[primary_span]
        span: Span,
        pat: String,
    },
}

#[derive(Diagnostic)]
#[diag("unexpected `||` before function parameter")]
#[note("alternatives in or-patterns are separated with `|`, not `||`")]
pub(crate) struct UnexpectedVertVertBeforeFunctionParam {
    #[primary_span]
    #[suggestion(label = "remove the `||`", code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token `||` in pattern")]
pub(crate) struct UnexpectedVertVertInPattern {
    #[primary_span]
    #[suggestion(
        label = "use a single `|` to separate multiple alternative patterns",
        code = "|",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    #[label("while parsing this or-pattern starting here")]
    pub start: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("a trailing `|` is not allowed in an or-pattern")]
pub(crate) struct TrailingVertNotAllowed {
    #[primary_span]
    #[suggestion(label = "remove the `{$token}`", code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label("while parsing this or-pattern starting here")]
    pub start: Option<Span>,
    pub token: Token,
    #[note("alternatives in or-patterns are separated with `|`, not `||`")]
    pub note_double_vert: Option<()>,
}

#[derive(Diagnostic)]
#[diag("unexpected `...`")]
pub(crate) struct DotDotDotRestPattern {
    #[primary_span]
    #[suggestion(
        label = "for a rest pattern, use `..` instead of `...`",
        style = "short",
        code = "..",
        applicability = "machine-applicable"
    )]
    #[label("not a valid pattern")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("pattern on wrong side of `@`")]
pub(crate) struct PatternOnWrongSideOfAt {
    #[primary_span]
    #[suggestion(
        label = "switch the order",
        code = "{whole_pat}",
        applicability = "machine-applicable"
    )]
    pub whole_span: Span,
    pub whole_pat: String,
    #[label("pattern on the left, should be on the right")]
    pub pattern: Span,
    #[label("binding on the right, should be on the left")]
    pub binding: Span,
}

#[derive(Diagnostic)]
#[diag("left-hand side of `@` must be a binding")]
#[note("bindings are `x`, `mut x`, `ref x`, and `ref mut x`")]
pub(crate) struct ExpectedBindingLeftOfAt {
    #[primary_span]
    pub whole_span: Span,
    #[label("interpreted as a pattern, not a binding")]
    pub lhs: Span,
    #[label("also a pattern")]
    pub rhs: Span,
}

#[derive(Diagnostic)]
#[diag("the range pattern here has ambiguous interpretation")]
pub(crate) struct AmbiguousRangePattern {
    #[primary_span]
    #[suggestion(
        label = "add parentheses to clarify the precedence",
        code = "({pat})",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
    pub pat: String,
}

#[derive(Diagnostic)]
#[diag("unexpected lifetime `{$symbol}` in pattern")]
pub(crate) struct UnexpectedLifetimeInPattern {
    #[primary_span]
    #[suggestion(label = "remove the lifetime", code = "", applicability = "machine-applicable")]
    pub span: Span,
    pub symbol: Symbol,
}

#[derive(Diagnostic)]
#[diag("the order of `mut` and `ref` is incorrect")]
pub(crate) struct RefMutOrderIncorrect {
    #[primary_span]
    #[suggestion(
        label = "try switching the order",
        code = "ref mut",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum InvalidMutInPattern {
    #[diag("`mut` must be attached to each individual binding")]
    #[note("`mut` may be followed by `variable` and `variable @ pattern`")]
    NestedIdent {
        #[primary_span]
        #[suggestion(
            label = "add `mut` to each binding",
            code = "{pat}",
            applicability = "machine-applicable"
        )]
        span: Span,
        pat: String,
    },
    #[diag("`mut` must be followed by a named binding")]
    #[note("`mut` may be followed by `variable` and `variable @ pattern`")]
    NonIdent {
        #[primary_span]
        #[suggestion(
            label = "remove the `mut` prefix",
            code = "",
            applicability = "machine-applicable"
        )]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("`mut` on a binding may not be repeated")]
pub(crate) struct RepeatedMutInPattern {
    #[primary_span]
    #[suggestion(
        label = "remove the additional `mut`s",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("range-to patterns with `...` are not allowed")]
pub(crate) struct DotDotDotRangeToPatternNotAllowed {
    #[primary_span]
    #[suggestion(
        label = "use `..=` instead",
        style = "short",
        code = "..=",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected identifier, found enum pattern")]
pub(crate) struct EnumPatternInsteadOfIdentifier {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected field pattern, found `{$token_str}`")]
pub(crate) struct DotDotDotForRemainingFields {
    #[primary_span]
    #[suggestion(
        label = "to omit remaining fields, use `..`",
        code = "..",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub token_str: Cow<'static, str>,
}

#[derive(Diagnostic)]
#[diag("expected `,`")]
pub(crate) struct ExpectedCommaAfterPatternField {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("return types are denoted using `->`")]
pub(crate) struct ReturnTypesUseThinArrow {
    #[primary_span]
    #[suggestion(
        label = "use `->` instead",
        style = "short",
        code = "->",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("lifetime in trait object type must be followed by `+`")]
pub(crate) struct NeedPlusAfterTraitObjectLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected `mut` or `const` keyword in raw pointer type")]
pub(crate) struct ExpectedMutOrConstInRawPointerType {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "add `mut` or `const` here",
        code("mut ", "const "),
        applicability = "has-placeholders"
    )]
    pub after_asterisk: Span,
}

#[derive(Diagnostic)]
#[diag("lifetime must precede `mut`")]
pub(crate) struct LifetimeAfterMut {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "place the lifetime before `mut`",
        code = "&{snippet} mut",
        applicability = "maybe-incorrect"
    )]
    pub suggest_lifetime: Option<Span>,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag("`mut` must precede `dyn`")]
pub(crate) struct DynAfterMut {
    #[primary_span]
    #[suggestion(
        label = "place `mut` before `dyn`",
        code = "&mut dyn",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("an `fn` pointer type cannot be `const`")]
pub(crate) struct FnPointerCannotBeConst {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "remove the `const` qualifier",
        code = "",
        applicability = "maybe-incorrect"
    )]
    #[label("`const` because of this")]
    pub qualifier: Span,
}

#[derive(Diagnostic)]
#[diag("an `fn` pointer type cannot be `async`")]
pub(crate) struct FnPointerCannotBeAsync {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "remove the `async` qualifier",
        code = "",
        applicability = "maybe-incorrect"
    )]
    #[label("`async` because of this")]
    pub qualifier: Span,
}

#[derive(Diagnostic)]
#[diag(label = "C-variadic type `...` may not be nested inside another type", code = "E0743")]
pub(crate) struct NestedCVariadicType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid `dyn` keyword")]
#[help("`dyn` is only needed at the start of a trait `+`-separated list")]
pub(crate) struct InvalidDynKeyword {
    #[primary_span]
    #[suggestion(label = "remove this keyword", code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub enum HelpUseLatestEdition {
    #[help("set `edition = \"{$edition}\"` in `Cargo.toml`")]
    #[note("for more on editions, read https://doc.rust-lang.org/edition-guide")]
    Cargo { edition: Edition },
    #[help("pass `--edition {$edition}` to `rustc`")]
    #[note("for more on editions, read https://doc.rust-lang.org/edition-guide")]
    Standalone { edition: Edition },
}

impl HelpUseLatestEdition {
    pub fn new() -> Self {
        let edition = LATEST_STABLE_EDITION;
        if std::env::var_os("CARGO").is_some() {
            Self::Cargo { edition }
        } else {
            Self::Standalone { edition }
        }
    }
}

#[derive(Diagnostic)]
#[diag("`box_syntax` has been removed")]
pub struct BoxSyntaxRemoved<'a> {
    #[primary_span]
    #[suggestion(
        label = "use `Box::new()` instead",
        code = "Box::new({code})",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
    pub code: &'a str,
}

#[derive(Diagnostic)]
#[diag("return type not allowed with return type notation")]
pub(crate) struct BadReturnTypeNotationOutput {
    #[primary_span]
    #[suggestion(label = "remove the return type", code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("return type notation uses `()` instead of `(..)` for elided arguments")]
pub(crate) struct BadReturnTypeNotationDotDot {
    #[primary_span]
    #[suggestion(label = "remove the `..`", code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("bounds on associated types do not belong here")]
pub(crate) struct BadAssocTypeBounds {
    #[primary_span]
    #[label("belongs in `where` clause")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("trailing attribute after generic parameter")]
pub(crate) struct AttrAfterGeneric {
    #[primary_span]
    #[label("attributes must go before parameters")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute without generic parameters")]
pub(crate) struct AttrWithoutGenerics {
    #[primary_span]
    #[label("attributes are only permitted when preceding parameters")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("generic parameters on `where` clauses are reserved for future use")]
pub(crate) struct WhereOnGenerics {
    #[primary_span]
    #[label("currently unsupported")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected generic arguments in path")]
pub(crate) struct GenericsInPath {
    #[primary_span]
    pub span: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("associated lifetimes are not supported")]
#[help("if you meant to specify a trait object, write `dyn Trait + 'lifetime`")]
pub(crate) struct AssocLifetime {
    #[primary_span]
    pub span: Span,
    #[label("the lifetime is given here")]
    pub lifetime: Span,
}

#[derive(Diagnostic)]
#[diag("`~const` may only modify trait bounds, not lifetime bounds")]
pub(crate) struct TildeConstLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$sigil}` may only modify trait bounds, not lifetime bounds")]
pub(crate) struct ModifierLifetime {
    #[primary_span]
    #[suggestion(
        label = "remove the `{$sigil}`",
        style = "tool-only",
        applicability = "maybe-incorrect",
        code = ""
    )]
    pub span: Span,
    pub modifier: &'static str,
}

#[derive(Diagnostic)]
#[diag("parenthesized lifetime bounds are not supported")]
pub(crate) struct ParenthesizedLifetime {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "remove the parentheses",
        style = "short",
        applicability = "machine-applicable",
        code = "{snippet}"
    )]
    pub sugg: Option<Span>,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag("const bounds must start with `~`")]
pub(crate) struct ConstMissingTilde {
    #[primary_span]
    pub span: Span,
    #[suggestion(label = "add `~`", code = "~", applicability = "machine-applicable")]
    pub start: Span,
}

#[derive(Diagnostic)]
#[diag("underscore literal suffix is not allowed")]
pub(crate) struct UnderscoreLiteralSuffix {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a label, found an identifier")]
pub(crate) struct ExpectedLabelFoundIdent {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "labels start with a tick",
        code = "'",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub start: Span,
}

#[derive(Diagnostic)]
#[diag("{$article} {$descr} cannot be `default`")]
#[note("only associated `fn`, `const`, and `type` items can be `default`")]
pub(crate) struct InappropriateDefault {
    #[primary_span]
    #[label("`default` because of this")]
    pub span: Span,
    pub article: &'static str,
    pub descr: &'static str,
}

#[derive(Diagnostic)]
#[diag("expected item, found {$token_name}")]
pub(crate) struct RecoverImportAsUse {
    #[primary_span]
    #[suggestion(
        label = "items are imported using the `use` keyword",
        code = "use",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub span: Span,
    pub token_name: String,
}

#[derive(Diagnostic)]
#[diag("expected `::`, found `:`")]
#[note("import paths are delimited using `::`")]
pub(crate) struct SingleColonImportPath {
    #[primary_span]
    #[suggestion(
        label = "use double colon",
        code = "::",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$descr} is not supported in {$ctx}")]
#[help("consider moving the {$descr} out to a nearby module scope")]
pub(crate) struct BadItemKind {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
    pub ctx: &'static str,
}

#[derive(Diagnostic)]
#[diag("found single colon in a struct field type path")]
pub(crate) struct SingleColonStructType {
    #[primary_span]
    #[suggestion(
        label = "write a path separator here",
        code = "::",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("default values on `struct` fields aren't supported")]
pub(crate) struct EqualsStructDefault {
    #[primary_span]
    #[suggestion(
        label = "remove this unsupported default value",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected `!` after `macro_rules`")]
pub(crate) struct MacroRulesMissingBang {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "add a `!`",
        code = "!",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("macro names aren't followed by a `!`")]
pub(crate) struct MacroNameRemoveBang {
    #[primary_span]
    #[suggestion(label = "remove the `!`", code = "", applicability = "machine-applicable")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("can't qualify macro_rules invocation with `{$vis}`")]
pub(crate) struct MacroRulesVisibility<'a> {
    #[primary_span]
    #[suggestion(
        label = "try exporting the macro",
        code = "#[macro_export]",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
    pub vis: &'a str,
}

#[derive(Diagnostic)]
#[diag("can't qualify macro invocation with `pub`")]
#[help("try adjusting the macro to put `{$vis}` inside the invocation")]
pub(crate) struct MacroInvocationVisibility<'a> {
    #[primary_span]
    #[suggestion(label = "remove the visibility", code = "", applicability = "machine-applicable")]
    pub span: Span,
    pub vis: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$kw_str}` definition cannot be nested inside `{$keyword}`")]
pub(crate) struct NestedAdt<'a> {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "consider creating a new `{$kw_str}` definition instead of nesting",
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub item: Span,
    pub keyword: &'a str,
    pub kw_str: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag("function body cannot be `= expression;`")]
pub(crate) struct FunctionBodyEqualsExpr {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: FunctionBodyEqualsExprSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "surround the expression with `{` and `}` instead of `=` and `;`",
    applicability = "machine-applicable"
)]
pub(crate) struct FunctionBodyEqualsExprSugg {
    #[suggestion_part(code = "{{")]
    pub eq: Span,
    #[suggestion_part(code = " }}")]
    pub semi: Span,
}

#[derive(Diagnostic)]
#[diag("expected pattern, found {$descr}")]
pub(crate) struct BoxNotPat {
    #[primary_span]
    pub span: Span,
    #[note("`box` is a reserved keyword")]
    pub kw: Span,
    #[suggestion(
        label = "escape `box` to use it as an identifier",
        code = "r#",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub lo: Span,
    pub descr: String,
}

#[derive(Diagnostic)]
#[diag(
    "unmatched angle {$plural ->
[true] brackets
*[false] bracket
}"
)]
pub(crate) struct UnmatchedAngle {
    #[primary_span]
    #[suggestion(
        label = "remove extra angle {$plural ->
[true] brackets
*[false] bracket
}",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub plural: bool,
}

#[derive(Diagnostic)]
#[diag("expected `+` between lifetime and {$sym}")]
pub(crate) struct MissingPlusBounds {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "add `+`",
        code = " +",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub hi: Span,
    pub sym: Symbol,
}

#[derive(Diagnostic)]
#[diag("incorrect parentheses around trait bounds")]
pub(crate) struct IncorrectParensTraitBounds {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: IncorrectParensTraitBoundsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(label = "fix the parentheses", applicability = "machine-applicable")]
pub(crate) struct IncorrectParensTraitBoundsSugg {
    #[suggestion_part(code = " ")]
    pub wrong_span: Span,
    #[suggestion_part(code = "(")]
    pub new_span: Span,
}

#[derive(Diagnostic)]
#[diag("keyword `{$kw}` is written in the wrong case")]
pub(crate) struct KwBadCase<'a> {
    #[primary_span]
    #[suggestion(
        label = "write it in the correct case",
        code = "{kw}",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub kw: &'a str,
}

#[derive(Diagnostic)]
#[diag("wrong meta list delimiters")]
pub(crate) struct MetaBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Diagnostic)]
#[diag("wrong `cfg_attr` delimiters")]
pub(crate) struct CfgAttrBadDelim {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: MetaBadDelimSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "the delimiters should be `(` and `)`",
    applicability = "machine-applicable"
)]
pub(crate) struct MetaBadDelimSugg {
    #[suggestion_part(code = "(")]
    pub open: Span,
    #[suggestion_part(code = ")")]
    pub close: Span,
}

#[derive(Diagnostic)]
#[diag("malformed `cfg_attr` attribute input")]
#[note(
    "for more information, visit <https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute>"
)]
pub(crate) struct MalformedCfgAttr {
    #[primary_span]
    #[suggestion(label = "missing condition and attribute", code = "{sugg}")]
    pub span: Span,
    pub sugg: &'static str,
}

#[derive(Diagnostic)]
#[diag("unknown `builtin #` construct `{$name}`")]
pub(crate) struct UnknownBuiltinConstruct {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("expected identifier after `builtin #`")]
pub(crate) struct ExpectedBuiltinIdent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("static items may not have generic parameters")]
pub(crate) struct StaticWithGenerics {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("where clauses are not allowed before const item bodies")]
pub(crate) struct WhereClauseBeforeConstBody {
    #[primary_span]
    #[label("unexpected where clause")]
    pub span: Span,
    #[label("while parsing this const item")]
    pub name: Span,
    #[label("the item body")]
    pub body: Span,
    #[subdiagnostic]
    pub sugg: Option<WhereClauseBeforeConstBodySugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    label = "move the body before the where clause",
    applicability = "machine-applicable"
)]
pub(crate) struct WhereClauseBeforeConstBodySugg {
    #[suggestion_part(code = "= {snippet} ")]
    pub left: Span,
    pub snippet: String,
    #[suggestion_part(code = "")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("generic args in patterns require the turbofish syntax")]
pub(crate) struct GenericArgsInPatRequireTurbofishSyntax {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        label = "use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments",
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
