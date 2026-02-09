// ignore-tidy-filelength

use std::borrow::Cow;
use std::path::PathBuf;

use rustc_ast::token::{self, InvisibleOrigin, MetaVarKind, Token};
use rustc_ast::util::parser::ExprPrecedence;
use rustc_ast::{Path, Visibility};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, IntoDiagArg,
    Level, Subdiagnostic, SuggestionStyle, inline_fluent,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::edition::{Edition, LATEST_STABLE_EDITION};
use rustc_span::{Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag("ambiguous `+` in a type")]
pub(crate) struct AmbiguousPlus {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: AddParen,
}

#[derive(Diagnostic)]
#[diag("expected a path on the left-hand side of `+`", code = E0178)]
pub(crate) struct BadTypePlus {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: BadTypePlusSub,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("try adding parentheses", applicability = "machine-applicable")]
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

#[derive(Diagnostic)]
#[diag("inherent impls cannot be {$modifier_name}")]
#[note("only trait implementations may be annotated with `{$modifier}`")]
pub(crate) struct TraitImplModifierInInherentImpl {
    #[primary_span]
    pub span: Span,
    pub modifier: &'static str,
    pub modifier_name: &'static str,
    #[label("{$modifier_name} because of this")]
    pub modifier_span: Span,
    #[label("inherent impl for this type")]
    pub self_ty: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "types that don't start with an identifier need to be surrounded with angle brackets in qualified paths",
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
        "remove this semicolon",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    #[help("{$name} declarations are not followed by a semicolon")]
    pub show_help: bool,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag("incorrect use of `await`")]
pub(crate) struct IncorrectUseOfAwait {
    #[primary_span]
    #[suggestion(
        "`await` is not a method call, remove the parentheses",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("incorrect use of `use`")]
pub(crate) struct IncorrectUseOfUse {
    #[primary_span]
    #[suggestion(
        "`use` is not a method call, try removing the parentheses",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("`await` is a postfix operation", applicability = "machine-applicable")]
pub(crate) struct AwaitSuggestion {
    #[suggestion_part(code = "")]
    pub removal: Span,
    #[suggestion_part(code = ".await{question_mark}")]
    pub dot_await: Span,
    pub question_mark: &'static str,
}

#[derive(Diagnostic)]
#[diag("incorrect use of `await`")]
pub(crate) struct IncorrectAwait {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: AwaitSuggestion,
}

#[derive(Diagnostic)]
#[diag("expected iterable, found keyword `in`")]
pub(crate) struct InInTypo {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove the duplicated `in`",
        code = "",
        style = "verbose",
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
        "switch the order of `mut` and `let`",
        style = "verbose",
        applicability = "maybe-incorrect",
        code = "let mut"
    )]
    SwitchMutLetOrder(#[primary_span] Span),
    #[suggestion(
        "missing keyword",
        applicability = "machine-applicable",
        style = "verbose",
        code = "let mut"
    )]
    MissingLet(#[primary_span] Span),
    #[suggestion(
        "write `let` instead of `auto` to introduce a new variable",
        style = "verbose",
        applicability = "machine-applicable",
        code = "let"
    )]
    UseLetNotAuto(#[primary_span] Span),
    #[suggestion(
        "write `let` instead of `var` to introduce a new variable",
        style = "verbose",
        applicability = "machine-applicable",
        code = "let"
    )]
    UseLetNotVar(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("switch the order of `ref` and `box`")]
pub(crate) struct SwitchRefBoxOrder {
    #[primary_span]
    #[suggestion(
        "swap them",
        applicability = "machine-applicable",
        style = "verbose",
        code = "box ref"
    )]
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
        "`{$invalid}` is not a valid comparison operator, use `{$correct}`",
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
        "use `&&` to perform logical conjunction",
        style = "verbose",
        applicability = "machine-applicable",
        code = "&&"
    )]
    Conjunction(#[primary_span] Span),
    #[suggestion(
        "use `||` to perform logical disjunction",
        style = "verbose",
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
        "use `!` to perform bitwise not",
        style = "verbose",
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
pub(crate) enum NotAsNegationOperatorSub {
    #[suggestion(
        "use `!` to perform logical negation or bitwise not",
        style = "verbose",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotDefault(#[primary_span] Span),

    #[suggestion(
        "use `!` to perform bitwise not",
        style = "verbose",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotBitwise(#[primary_span] Span),

    #[suggestion(
        "use `!` to perform logical negation",
        style = "verbose",
        applicability = "machine-applicable",
        code = "!"
    )]
    SuggestNotLogical(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("malformed loop label")]
pub(crate) struct MalformedLoopLabel {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use the correct loop label format",
        applicability = "machine-applicable",
        code = "'",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("borrow expressions cannot be annotated with lifetimes")]
pub(crate) struct LifetimeInBorrowExpression {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove the lifetime annotation",
        applicability = "machine-applicable",
        code = "",
        style = "verbose"
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
#[diag("expected `while`, `for`, `loop` or `{\"{\"}` after a label")]
pub(crate) struct UnexpectedTokenAfterLabel {
    #[primary_span]
    #[label("expected `while`, `for`, `loop` or `{\"{\"}` after a label")]
    pub span: Span,
    #[suggestion("consider removing the label", style = "verbose", code = "")]
    pub remove_label: Option<Span>,
    #[subdiagnostic]
    pub enclose_in_block: Option<UnexpectedTokenAfterLabelSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider enclosing expression in a block",
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
        "add `:` after the label",
        style = "verbose",
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
        "replace with the new syntax",
        applicability = "machine-applicable",
        code = "try",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("float literals must have an integer part")]
pub(crate) struct FloatLiteralRequiresIntegerPart {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "must have an integer part",
        applicability = "machine-applicable",
        code = "0",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("expected `;`, found `[`")]
pub(crate) struct MissingSemicolonBeforeArray {
    #[primary_span]
    pub open_delim: Span,
    #[suggestion(
        "consider adding `;` here",
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
        "use `..` to fill in the rest of the fields",
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
#[multipart_suggestion("wrap this in another block", applicability = "machine-applicable")]
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
pub(crate) struct TernaryOperator {
    #[primary_span]
    pub span: Span,
    /// If we have a span for the condition expression, suggest the if/else
    #[subdiagnostic]
    pub sugg: Option<TernaryOperatorSuggestion>,
    /// Otherwise, just print the suggestion message
    #[help("use an `if-else` expression instead")]
    pub no_sugg: bool,
}

#[derive(Subdiagnostic, Copy, Clone)]
#[multipart_suggestion(
    "use an `if-else` expression instead",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct TernaryOperatorSuggestion {
    #[suggestion_part(code = "if ")]
    pub before_cond: Span,
    #[suggestion_part(code = "{{")]
    pub question: Span,
    #[suggestion_part(code = "}} else {{")]
    pub colon: Span,
    #[suggestion_part(code = " }}")]
    pub end: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "remove the `if` if you meant to write a `let...else` statement",
    applicability = "maybe-incorrect",
    code = "",
    style = "verbose"
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

#[derive(Diagnostic)]
#[diag("let-chain with missing `let`")]
pub(crate) struct LetChainMissingLet {
    #[primary_span]
    pub span: Span,
    #[label("expected `let` expression, found assignment")]
    pub label_span: Span,
    #[label("let expression later in the condition")]
    pub rhs_span: Span,
    #[suggestion(
        "add `let` before the expression",
        applicability = "maybe-incorrect",
        code = "let ",
        style = "verbose"
    )]
    pub sug_span: Span,
}

#[derive(Diagnostic)]
#[diag("`||` operators are not supported in let chain conditions")]
pub(crate) struct OrInLetChain {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic, Clone, Copy)]
#[multipart_suggestion(
    "you might have meant to continue the let-chain",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct MaybeMissingLet {
    #[suggestion_part(code = "let ")]
    pub span: Span,
}

#[derive(Subdiagnostic, Clone, Copy)]
#[multipart_suggestion(
    "you might have meant to compare for equality",
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
        "consider using `=` here",
        applicability = "maybe-incorrect",
        code = "=",
        style = "verbose"
    )]
    pub sugg_span: Span,
}

#[derive(Diagnostic)]
#[diag("expected `{\"{\"}`, found {$first_tok}")]
pub(crate) struct ExpectedElseBlock {
    #[primary_span]
    pub first_tok_span: Span,
    pub first_tok: String,
    #[label("expected an `if` or a block after this `else`")]
    pub else_span: Span,
    #[suggestion(
        "add an `if` if this is the condition of a chained `else if` statement",
        applicability = "maybe-incorrect",
        code = "if ",
        style = "verbose"
    )]
    pub condition_start: Span,
}

#[derive(Diagnostic)]
#[diag("expected one of `,`, `:`, or `{\"}\"}`, found `{$token}`")]
pub(crate) struct ExpectedStructField {
    #[primary_span]
    #[label("expected one of `,`, `:`, or `{\"}\"}`")]
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

    #[suggestion(
        "remove the attributes",
        applicability = "machine-applicable",
        code = "",
        style = "verbose"
    )]
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
    // User wrote `for pat of expr {}`
    // Has been misleading, at least in the past (closed Issue #48492), thus maybe-incorrect
    #[suggestion(
        "try using `in` here instead",
        style = "verbose",
        applicability = "maybe-incorrect",
        code = "in"
    )]
    InNotOf(#[primary_span] Span),
    // User wrote `for pat = expr {}`
    #[suggestion(
        "try using `in` here instead",
        style = "verbose",
        applicability = "maybe-incorrect",
        code = "in"
    )]
    InNotEq(#[primary_span] Span),
    #[suggestion(
        "try adding `in` here",
        style = "verbose",
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
        "try adding an expression to the `for` loop",
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
        "missing a comma here to end this `match` arm",
        applicability = "machine-applicable",
        code = ",",
        style = "verbose"
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
#[diag("cannot use a comma after the base struct")]
#[note("the base struct must always be the last field")]
pub(crate) struct CommaAfterBaseStruct {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove this comma",
        style = "verbose",
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
        "replace equals symbol with a colon",
        applicability = "machine-applicable",
        code = ":",
        style = "verbose"
    )]
    pub eq: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token: `...`")]
pub(crate) struct DotDotDot {
    #[primary_span]
    #[suggestion(
        "use `..` for an exclusive range",
        applicability = "maybe-incorrect",
        code = "..",
        style = "verbose"
    )]
    #[suggestion(
        "or `..=` for an inclusive range",
        applicability = "maybe-incorrect",
        code = "..=",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token: `<-`")]
pub(crate) struct LeftArrowOperator {
    #[primary_span]
    #[suggestion(
        "if you meant to write a comparison against a negative value, add a space in between `<` and `-`",
        applicability = "maybe-incorrect",
        code = "< -",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected pattern, found `let`")]
pub(crate) struct RemoveLet {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove the unnecessary `let` keyword",
        applicability = "machine-applicable",
        code = "",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `==`")]
pub(crate) struct UseEqInstead {
    #[primary_span]
    #[suggestion(
        "try using `=` instead",
        style = "verbose",
        applicability = "machine-applicable",
        code = "="
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected { \"`{}`\" }, found `;`")]
pub(crate) struct UseEmptyBlockNotSemi {
    #[primary_span]
    #[suggestion(
        r#"try using { "`{}`" } instead"#,
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
    pub suggestion: ComparisonInterpretedAsGenericSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("try comparing the cast value", applicability = "machine-applicable")]
pub(crate) struct ComparisonInterpretedAsGenericSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
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
    pub suggestion: ShiftInterpretedAsGenericSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("try shifting the cast value", applicability = "machine-applicable")]
pub(crate) struct ShiftInterpretedAsGenericSugg {
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
#[diag("extra characters after frontmatter close are not allowed")]
pub(crate) struct FrontmatterExtraCharactersAfterClose {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid infostring for frontmatter")]
#[note("frontmatter infostrings must be a single identifier immediately following the opening")]
pub(crate) struct FrontmatterInvalidInfostring {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid preceding whitespace for frontmatter opening")]
pub(crate) struct FrontmatterInvalidOpeningPrecedingWhitespace {
    #[primary_span]
    pub span: Span,
    #[note("frontmatter opening should not be preceded by whitespace")]
    pub note_span: Span,
}

#[derive(Diagnostic)]
#[diag("unclosed frontmatter")]
pub(crate) struct FrontmatterUnclosed {
    #[primary_span]
    pub span: Span,
    #[note("frontmatter opening here was not closed")]
    pub note_span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid preceding whitespace for frontmatter close")]
pub(crate) struct FrontmatterInvalidClosingPrecedingWhitespace {
    #[primary_span]
    pub span: Span,
    #[note("frontmatter close should not be preceded by whitespace")]
    pub note_span: Span,
}

#[derive(Diagnostic)]
#[diag("frontmatter close does not match the opening")]
pub(crate) struct FrontmatterLengthMismatch {
    #[primary_span]
    pub span: Span,
    #[label("the opening here has {$len_opening} dashes...")]
    pub opening: Span,
    #[label("...while the close has {$len_close} dashes")]
    pub close: Span,
    pub len_opening: usize,
    pub len_close: usize,
}

#[derive(Diagnostic)]
#[diag(
    "too many `-` symbols: frontmatter openings may be delimited by up to 255 `-` symbols, but found {$len_opening}"
)]
pub(crate) struct FrontmatterTooManyDashes {
    pub len_opening: usize,
}

#[derive(Diagnostic)]
#[diag("bare CR not allowed in frontmatter")]
pub(crate) struct BareCrFrontmatter {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("leading `+` is not supported")]
pub(crate) struct LeadingPlusNotSupported {
    #[primary_span]
    #[label("unexpected `+`")]
    pub span: Span,
    #[suggestion(
        "try removing the `+`",
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
    "if `{$type}` is a struct, use braces as delimiters",
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
    "if `{$type}` is a function, use the arguments directly",
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
    pub sub: WrapInParentheses,
}

#[derive(Subdiagnostic)]
pub(crate) enum WrapInParentheses {
    #[multipart_suggestion(
        "wrap the expression in parentheses",
        applicability = "machine-applicable"
    )]
    Expression {
        #[suggestion_part(code = "(")]
        left: Span,
        #[suggestion_part(code = ")")]
        right: Span,
    },
    #[multipart_suggestion(
        "use parentheses instead of braces for this macro",
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
#[diag("this is a block expression, not an array")]
pub(crate) struct ArrayBracketsInsteadOfBraces {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: ArrayBracketsInsteadOfBracesSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "to make an array, use square brackets instead of curly braces",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ArrayBracketsInsteadOfBracesSugg {
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
        "use `..=` instead",
        style = "verbose",
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
        "add a space between the pattern and `=>`",
        style = "verbose",
        code = " ",
        applicability = "machine-applicable"
    )]
    pub after_pat: Span,
}

#[derive(Diagnostic)]
#[diag("inclusive range with no end", code = E0586)]
#[note("inclusive ranges must be bounded at the end (`..=b` or `a..=b`)")]
pub(crate) struct InclusiveRangeNoEnd {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use `..` instead",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum MatchArmBodyWithoutBracesSugg {
    #[multipart_suggestion(
        "surround the {$num_statements ->
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
        "replace `;` with `,` to end a `match` arm expression",
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
#[diag("struct literals are not allowed here")]
pub(crate) struct StructLiteralNotAllowedHere {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: StructLiteralNotAllowedHereSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "surround the struct literal with parentheses",
    applicability = "machine-applicable"
)]
pub(crate) struct StructLiteralNotAllowedHereSugg {
    #[suggestion_part(code = "(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("suffixes on a tuple index are invalid")]
pub(crate) struct InvalidLiteralSuffixOnTupleIndex {
    #[primary_span]
    #[label("invalid suffix `{$suffix}`")]
    pub span: Span,
    pub suffix: Symbol,
}

#[derive(Diagnostic)]
#[diag("non-string ABI literal")]
pub(crate) struct NonStringAbiLiteral {
    #[primary_span]
    #[suggestion(
        "specify the ABI with a string literal",
        code = "\"C\"",
        applicability = "maybe-incorrect",
        style = "verbose"
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
#[diag("incorrect visibility restriction", code = E0704)]
#[help(
    "some possible visibility restrictions are:
    `pub(crate)`: visible only on the current crate
    `pub(super)`: visible only in the current module's parent
    `pub(in path::to::module)`: visible only on the specified path"
)]
pub(crate) struct IncorrectVisibilityRestriction {
    #[primary_span]
    #[suggestion(
        "make this visible only to module `{$inner_str}` with `in`",
        code = "in {inner_str}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
    pub inner_str: String,
}

#[derive(Diagnostic)]
#[diag("<assignment> ... else {\"{\"} ... {\"}\"} is not allowed")]
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
#[diag("found a documentation comment that doesn't document anything", code = E0585)]
#[help("doc comments must come before what they document, if a comment was intended use `//`")]
pub(crate) struct DocCommentDoesNotDocumentAnything {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "missing comma here",
        code = ",",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub missing_comma: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("`const` and `let` are mutually exclusive")]
pub(crate) struct ConstLetMutuallyExclusive {
    #[primary_span]
    #[suggestion(
        "remove `let`",
        code = "const",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("a `{$operator}` expression cannot be directly assigned in `let...else`")]
pub(crate) struct InvalidExpressionInLetElse {
    #[primary_span]
    pub span: Span,
    pub operator: &'static str,
    #[subdiagnostic]
    pub sugg: WrapInParentheses,
}

#[derive(Diagnostic)]
#[diag("right curly brace `{\"}\"}` before `else` in a `let...else` statement not allowed")]
pub(crate) struct InvalidCurlyInLetElse {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WrapInParentheses,
}

#[derive(Diagnostic)]
#[diag("can't reassign to an uninitialized variable")]
#[help("if you meant to overwrite, remove the `let` binding")]
pub(crate) struct CompoundAssignmentExpressionInLet {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "initialize the variable",
        style = "verbose",
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub suggestion: Span,
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
#[diag("expected unsuffixed literal, found {$descr}")]
pub(crate) struct InvalidMetaItem {
    #[primary_span]
    pub span: Span,
    pub descr: String,
    #[subdiagnostic]
    pub quote_ident_sugg: Option<InvalidMetaItemQuoteIdentSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "surround the identifier with quotation marks to make it into a string literal",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidMetaItemQuoteIdentSugg {
    #[suggestion_part(code = "\"")]
    pub before: Span,
    #[suggestion_part(code = "\"")]
    pub after: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "escape `{$ident_name}` to use it as an identifier",
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
    "remove this comma",
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
    "you might have meant to introduce a new binding",
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
    #[label("expected identifier, found metavariable")]
    MetaVar(#[primary_span] Span),
    #[label("expected identifier")]
    Other(#[primary_span] Span),
}

impl ExpectedIdentifierFound {
    pub(crate) fn new(token_descr: Option<TokenDescription>, span: Span) -> Self {
        (match token_descr {
            Some(TokenDescription::ReservedIdentifier) => {
                ExpectedIdentifierFound::ReservedIdentifier
            }
            Some(TokenDescription::Keyword) => ExpectedIdentifierFound::Keyword,
            Some(TokenDescription::ReservedKeyword) => ExpectedIdentifierFound::ReservedKeyword,
            Some(TokenDescription::DocComment) => ExpectedIdentifierFound::DocComment,
            Some(TokenDescription::MetaVar(_)) => ExpectedIdentifierFound::MetaVar,
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

        let mut add_token = true;
        let mut diag = Diag::new(
            dcx,
            level,
            match token_descr {
                Some(TokenDescription::ReservedIdentifier) => {
                    inline_fluent!("expected identifier, found reserved identifier `{$token}`")
                }
                Some(TokenDescription::Keyword) => {
                    inline_fluent!("expected identifier, found keyword `{$token}`")
                }
                Some(TokenDescription::ReservedKeyword) => {
                    inline_fluent!("expected identifier, found reserved keyword `{$token}`")
                }
                Some(TokenDescription::DocComment) => {
                    inline_fluent!("expected identifier, found doc comment `{$token}`")
                }
                Some(TokenDescription::MetaVar(_)) => {
                    add_token = false;
                    inline_fluent!("expected identifier, found metavariable")
                }
                None => inline_fluent!("expected identifier, found `{$token}`"),
            },
        );
        diag.span(self.span);
        if add_token {
            diag.arg("token", self.token);
        }

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

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for ExpectedSemi {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let token_descr = TokenDescription::from_token(&self.token);

        let mut add_token = true;
        let mut diag = Diag::new(
            dcx,
            level,
            match token_descr {
                Some(TokenDescription::ReservedIdentifier) => {
                    inline_fluent!("expected `;`, found reserved identifier `{$token}`")
                }
                Some(TokenDescription::Keyword) => {
                    inline_fluent!("expected `;`, found keyword `{$token}`")
                }
                Some(TokenDescription::ReservedKeyword) => {
                    inline_fluent!("expected `;`, found reserved keyword `{$token}`")
                }
                Some(TokenDescription::DocComment) => {
                    inline_fluent!("expected `;`, found doc comment `{$token}`")
                }
                Some(TokenDescription::MetaVar(_)) => {
                    add_token = false;
                    inline_fluent!("expected `;`, found metavariable")
                }
                None => inline_fluent!("expected `;`, found `{$token}`"),
            },
        );
        diag.span(self.span);
        if add_token {
            diag.arg("token", self.token);
        }

        if let Some(unexpected_token_label) = self.unexpected_token_label {
            diag.span_label(unexpected_token_label, inline_fluent!("unexpected token"));
        }

        self.sugg.add_to_diag(&mut diag);

        diag
    }
}

#[derive(Subdiagnostic)]
pub(crate) enum ExpectedSemiSugg {
    #[suggestion(
        "change this to `;`",
        code = ";",
        applicability = "machine-applicable",
        style = "short"
    )]
    ChangeToSemi(#[primary_span] Span),
    #[suggestion("add `;` here", code = ";", applicability = "machine-applicable", style = "short")]
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
    "you might have forgotten to add the struct literal inside the block",
    applicability = "has-placeholders"
)]
pub(crate) struct StructLiteralBodyWithoutPathSugg {
    #[suggestion_part(code = "{{ SomeStruct ")]
    pub before: Span,
    #[suggestion_part(code = " }}")]
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
        "{$num_extra_brackets ->
            [one] remove extra angle bracket
            *[other] remove extra angle brackets
        }",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
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
    "surround the type parameters with angle brackets",
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
        "use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments",
        style = "verbose",
        code = "::",
        applicability = "maybe-incorrect"
    )]
    pub suggest_turbofish: Option<Span>,
    #[help("use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments")]
    #[help("or use `(...)` if you meant to specify fn arguments")]
    pub help_turbofish: bool,
    #[subdiagnostic]
    pub chaining_sugg: Option<ComparisonOperatorsCannotBeChainedSugg>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ComparisonOperatorsCannotBeChainedSugg {
    #[suggestion(
        "split the comparison into two",
        style = "verbose",
        code = " && {middle_term}",
        applicability = "maybe-incorrect"
    )]
    SplitComparison {
        #[primary_span]
        span: Span,
        middle_term: String,
    },
    #[multipart_suggestion("parenthesize the comparison", applicability = "maybe-incorrect")]
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
    "if you meant to express that the type might not contain a value, use the `Option` wrapper type",
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
#[multipart_suggestion("remove parentheses in `for` loop", applicability = "machine-applicable")]
pub(crate) struct ParenthesesInForHeadSugg {
    #[suggestion_part(code = " ")]
    pub left: Span,
    #[suggestion_part(code = " ")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected parentheses surrounding `match` arm pattern")]
pub(crate) struct ParenthesesInMatchPat {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: ParenthesesInMatchPatSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "remove parentheses surrounding the pattern",
    applicability = "machine-applicable"
)]
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
#[diag("attributes cannot be applied to types")]
pub(crate) struct AttributeOnType {
    #[primary_span]
    #[label("attributes are not allowed here")]
    pub span: Span,
    #[suggestion(
        "remove attribute from here",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub fix_span: Span,
}

#[derive(Diagnostic)]
#[diag("attributes cannot be applied to generic arguments")]
pub(crate) struct AttributeOnGenericArg {
    #[primary_span]
    #[label("attributes are not allowed here")]
    pub span: Span,
    #[suggestion(
        "remove attribute from here",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub fix_span: Span,
}

#[derive(Diagnostic)]
#[diag("attributes cannot be applied here")]
pub(crate) struct AttributeOnEmptyType {
    #[primary_span]
    #[label("attributes are not allowed here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("patterns aren't allowed in methods without bodies", code = E0642)]
pub(crate) struct PatternMethodParamWithoutBody {
    #[primary_span]
    #[suggestion(
        "give this argument a name or use an underscore to ignore it",
        code = "_",
        applicability = "machine-applicable",
        style = "verbose"
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
    "enclose the `const` expression in braces",
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
        "`const` parameters must be declared for the `impl`",
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
        "`const` parameters must be declared for the `impl`",
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
        "the `const` keyword is only needed in the definition of the type",
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
        "try switching the order",
        style = "verbose",
        code = "async move",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the order of `use` and `async` is incorrect")]
pub(crate) struct AsyncUseOrderIncorrect {
    #[primary_span]
    #[suggestion(
        "try switching the order",
        style = "verbose",
        code = "async use",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected `:` followed by trait or lifetime")]
pub(crate) struct DoubleColonInBound {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use single colon",
        code = ": ",
        applicability = "machine-applicable",
        style = "verbose"
    )]
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
    "place the return type after the function parameters",
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
#[multipart_suggestion(
    "consider moving the lifetime {$arity ->
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
}

impl Subdiagnostic for FnTraitMissingParen {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_label(self.span, inline_fluent!("`Fn` bounds require arguments in parentheses"));
        diag.span_suggestion_short(
            self.span.shrink_to_hi(),
            inline_fluent!("try adding parentheses"),
            "()",
            Applicability::MachineApplicable,
        );
    }
}

#[derive(Diagnostic)]
#[diag("unexpected `if` in the condition expression")]
pub(crate) struct UnexpectedIfWithIf(
    #[primary_span]
    #[suggestion(
        "remove the `if`",
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
        "replace `fn` with `impl` here",
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
        "use `Fn` to refer to the trait",
        applicability = "machine-applicable",
        code = "Fn",
        style = "verbose"
    )]
    pub fn_token_span: Span,
}

#[derive(Diagnostic)]
#[diag("`Trait(...)` syntax does not support named parameters")]
pub(crate) struct FnPathFoundNamedParams {
    #[primary_span]
    #[suggestion("remove the parameter name", applicability = "machine-applicable", code = "")]
    pub named_param_span: Span,
}

#[derive(Diagnostic)]
#[diag("`Trait(...)` syntax does not support c_variadic parameters")]
pub(crate) struct PathFoundCVariadicParams {
    #[primary_span]
    #[suggestion("remove the `...`", applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`Trait(...)` syntax does not support attributes in parameters")]
pub(crate) struct PathFoundAttributeInParams {
    #[primary_span]
    #[suggestion("remove the attributes", applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("path separator must be a double colon")]
pub(crate) struct PathSingleColon {
    #[primary_span]
    pub span: Span,

    #[suggestion(
        "use a double colon instead",
        applicability = "machine-applicable",
        code = ":",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("path separator must be a double colon")]
pub(crate) struct PathTripleColon {
    #[primary_span]
    #[suggestion(
        "use a double colon instead",
        applicability = "maybe-incorrect",
        code = "",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("statements are terminated with a semicolon")]
pub(crate) struct ColonAsSemi {
    #[primary_span]
    #[suggestion(
        "use a semicolon instead",
        applicability = "machine-applicable",
        code = ";",
        style = "verbose"
    )]
    pub span: Span,
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
    "move the body before the where clause",
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
#[diag("`async fn` is not permitted in Rust 2015", code = E0670)]
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
#[diag("`async use` blocks are only allowed in Rust 2018 or later")]
pub(crate) struct AsyncUseBlockIn2015 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`async` trait bounds are only allowed in Rust 2018 or later")]
pub(crate) struct AsyncBoundModifierIn2015 {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub help: HelpUseLatestEdition,
}

#[derive(Diagnostic)]
#[diag("let chains are only allowed in Rust 2024 or later")]
pub(crate) struct LetChainPre2024 {
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
#[diag("unexpected token: {$actual}")]
pub(crate) struct UnexpectedTokenAfterDot {
    #[primary_span]
    pub span: Span,
    pub actual: String,
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
    #[diag("missing `enum` for enum definition")]
    Enum {
        #[primary_span]
        span: Span,
        #[suggestion(
            "add `enum` here to parse `{$ident}` as an enum",
            style = "verbose",
            applicability = "maybe-incorrect",
            code = "enum "
        )]
        insert_span: Span,
        ident: Ident,
    },
    #[diag("missing `enum` or `struct` for enum or struct definition")]
    EnumOrStruct {
        #[primary_span]
        span: Span,
    },
    #[diag("missing `struct` for struct definition")]
    Struct {
        #[primary_span]
        span: Span,
        #[suggestion(
            "add `struct` here to parse `{$ident}` as a struct",
            style = "verbose",
            applicability = "maybe-incorrect",
            code = "struct "
        )]
        insert_span: Span,
        ident: Ident,
    },
    #[diag("missing `fn` for function definition")]
    Function {
        #[primary_span]
        span: Span,
        #[suggestion(
            "add `fn` here to parse `{$ident}` as a function",
            style = "verbose",
            applicability = "maybe-incorrect",
            code = "fn "
        )]
        insert_span: Span,
        ident: Ident,
    },
    #[diag("missing `fn` for method definition")]
    Method {
        #[primary_span]
        span: Span,
        #[suggestion(
            "add `fn` here to parse `{$ident}` as a method",
            style = "verbose",
            applicability = "maybe-incorrect",
            code = "fn "
        )]
        insert_span: Span,
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
        "if you meant to call a macro, try",
        applicability = "maybe-incorrect",
        code = "{snippet}!",
        style = "verbose"
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
        "add a parameter list",
        code = "()",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid path separator in function definition")]
pub(crate) struct InvalidPathSepInFnDefinition {
    #[primary_span]
    #[suggestion(
        "remove invalid path separator",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("missing trait in a trait impl")]
pub(crate) struct MissingTraitInTraitImpl {
    #[primary_span]
    #[suggestion(
        "add a trait here",
        code = " Trait ",
        applicability = "has-placeholders",
        style = "verbose"
    )]
    pub span: Span,
    #[suggestion(
        "for an inherent impl, drop this `for`",
        code = "",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub for_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing `for` in a trait impl")]
pub(crate) struct MissingForInTraitImpl {
    #[primary_span]
    #[suggestion(
        "add `for` here",
        style = "verbose",
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
    #[suggestion(
        "remove the extra `impl`",
        code = "",
        applicability = "maybe-incorrect",
        style = "short"
    )]
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
    "if the original crate name uses dashes you need to use underscores in the code",
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
        "try using a static value",
        code = "static ",
        applicability = "machine-applicable",
        style = "verbose"
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
        "you might want to declare a static instead",
        code = "static",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    pub const_span: Span,
}

#[derive(Diagnostic)]
#[diag("missing type for `{$kind}` item")]
pub(crate) struct MissingConstType {
    #[primary_span]
    #[suggestion(
        "provide a type for the item",
        code = "{colon} <type>",
        style = "verbose",
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
        "replace `enum struct` with",
        code = "enum",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum UnexpectedTokenAfterStructName {
    #[diag(
        "expected `where`, `{\"{\"}`, `(`, or `;` after struct name, found reserved identifier `{$token}`"
    )]
    ReservedIdentifier {
        #[primary_span]
        #[label("expected `where`, `{\"{\"}`, `(`, or `;` after struct name")]
        span: Span,
        token: Token,
    },
    #[diag("expected `where`, `{\"{\"}`, `(`, or `;` after struct name, found keyword `{$token}`")]
    Keyword {
        #[primary_span]
        #[label("expected `where`, `{\"{\"}`, `(`, or `;` after struct name")]
        span: Span,
        token: Token,
    },
    #[diag(
        "expected `where`, `{\"{\"}`, `(`, or `;` after struct name, found reserved keyword `{$token}`"
    )]
    ReservedKeyword {
        #[primary_span]
        #[label("expected `where`, `{\"{\"}`, `(`, or `;` after struct name")]
        span: Span,
        token: Token,
    },
    #[diag(
        "expected `where`, `{\"{\"}`, `(`, or `;` after struct name, found doc comment `{$token}`"
    )]
    DocComment {
        #[primary_span]
        #[label("expected `where`, `{\"{\"}`, `(`, or `;` after struct name")]
        span: Span,
        token: Token,
    },
    #[diag("expected `where`, `{\"{\"}`, `(`, or `;` after struct name, found metavar")]
    MetaVar {
        #[primary_span]
        #[label("expected `where`, `{\"{\"}`, `(`, or `;` after struct name")]
        span: Span,
    },
    #[diag("expected `where`, `{\"{\"}`, `(`, or `;` after struct name, found `{$token}`")]
    Other {
        #[primary_span]
        #[label("expected `where`, `{\"{\"}`, `(`, or `;` after struct name")]
        span: Span,
        token: Token,
    },
}

impl UnexpectedTokenAfterStructName {
    pub(crate) fn new(span: Span, token: Token) -> Self {
        match TokenDescription::from_token(&token) {
            Some(TokenDescription::ReservedIdentifier) => Self::ReservedIdentifier { span, token },
            Some(TokenDescription::Keyword) => Self::Keyword { span, token },
            Some(TokenDescription::ReservedKeyword) => Self::ReservedKeyword { span, token },
            Some(TokenDescription::DocComment) => Self::DocComment { span, token },
            Some(TokenDescription::MetaVar(_)) => Self::MetaVar { span },
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
        "consider joining the two `where` clauses into one",
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
    #[diag("`let` bindings require top-level or-patterns in parentheses")]
    LetBinding {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        sub: Option<TopLevelOrPatternNotAllowedSugg>,
    },
    #[diag("function parameters require top-level or-patterns in parentheses")]
    FunctionParameter {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        sub: Option<TopLevelOrPatternNotAllowedSugg>,
    },
}

#[derive(Diagnostic)]
#[diag("`{$ident}` cannot be a raw identifier")]
pub(crate) struct CannotBeRawIdent {
    #[primary_span]
    pub span: Span,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag("`{$ident}` cannot be a raw lifetime")]
pub(crate) struct CannotBeRawLifetime {
    #[primary_span]
    pub span: Span,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag("lifetimes cannot use keyword names")]
pub(crate) struct KeywordLifetime {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("labels cannot use keyword names")]
pub(crate) struct KeywordLabel {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "bare CR not allowed in {$block ->
        [true] block doc-comment
        *[false] doc-comment
    }"
)]
pub(crate) struct CrDocComment {
    #[primary_span]
    pub span: Span,
    pub block: bool,
}

#[derive(Diagnostic)]
#[diag("no valid digits found for number", code = E0768)]
pub(crate) struct NoDigitsLiteral {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid digit for a base {$base} literal")]
pub(crate) struct InvalidDigitLiteral {
    #[primary_span]
    pub span: Span,
    pub base: u32,
}

#[derive(Diagnostic)]
#[diag("expected at least one digit in exponent")]
pub(crate) struct EmptyExponentFloat {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$base} float literal is not supported")]
pub(crate) struct FloatLiteralUnsupportedBase {
    #[primary_span]
    pub span: Span,
    pub base: &'static str,
}

#[derive(Diagnostic)]
#[diag("prefix `{$prefix}` is unknown")]
#[note("prefixed identifiers and literals are reserved since Rust 2021")]
pub(crate) struct UnknownPrefix<'a> {
    #[primary_span]
    #[label("unknown prefix")]
    pub span: Span,
    pub prefix: &'a str,
    #[subdiagnostic]
    pub sugg: Option<UnknownPrefixSugg>,
}

#[derive(Subdiagnostic)]
#[note("macros cannot expand to {$adt_ty} fields")]
pub(crate) struct MacroExpandsToAdtField<'a> {
    pub adt_ty: &'a str,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnknownPrefixSugg {
    #[suggestion(
        "use `br` for a raw byte string",
        code = "br",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    UseBr(#[primary_span] Span),
    #[suggestion(
        "use `cr` for a raw C-string",
        code = "cr",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    UseCr(#[primary_span] Span),
    #[suggestion(
        "consider inserting whitespace here",
        code = " ",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Whitespace(#[primary_span] Span),
    #[multipart_suggestion(
        "if you meant to write a string literal, use double quotes",
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
#[diag("reserved multi-hash token is forbidden")]
#[note("sequences of two or more # are reserved for future use since Rust 2024")]
pub(crate) struct ReservedMultihash {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<GuardedStringSugg>,
}
#[derive(Diagnostic)]
#[diag("invalid string literal")]
#[note("unprefixed guarded string literals are reserved for future use since Rust 2024")]
pub(crate) struct ReservedString {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: Option<GuardedStringSugg>,
}
#[derive(Subdiagnostic)]
#[suggestion(
    "consider inserting whitespace here",
    code = " ",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct GuardedStringSugg(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag(
    "too many `#` symbols: raw strings may be delimited by up to 255 `#` symbols, but found {$num}"
)]
pub(crate) struct TooManyHashes {
    #[primary_span]
    pub span: Span,
    pub num: u32,
}

#[derive(Diagnostic)]
#[diag("unknown start of token: {$escaped}")]
pub(crate) struct UnknownTokenStart {
    #[primary_span]
    pub span: Span,
    pub escaped: String,
    #[subdiagnostic]
    pub sugg: Option<TokenSubstitution>,
    #[subdiagnostic]
    pub null: Option<UnknownTokenNull>,
    #[subdiagnostic]
    pub repeat: Option<UnknownTokenRepeat>,
    #[subdiagnostic]
    pub invisible: Option<InvisibleCharacter>,
}

#[derive(Subdiagnostic)]
pub(crate) enum TokenSubstitution {
    #[suggestion(
        "Unicode characters '“' (Left Double Quotation Mark) and '”' (Right Double Quotation Mark) look like '{$ascii_str}' ({$ascii_name}), but are not",
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
        "Unicode character '{$ch}' ({$u_name}) looks like '{$ascii_str}' ({$ascii_name}), but it is not",
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
#[note(
    "character appears {$repeats ->
        [one] once more
        *[other] {$repeats} more times
    }"
)]
pub(crate) struct UnknownTokenRepeat {
    pub repeats: usize,
}

#[derive(Subdiagnostic)]
#[help("invisible characters like '{$escaped}' are not usually visible in text editors")]
pub(crate) struct InvisibleCharacter;

#[derive(Subdiagnostic)]
#[help(
    "source files must contain UTF-8 encoded text, unexpected null bytes might occur when a different encoding is used"
)]
pub(crate) struct UnknownTokenNull;

#[derive(Diagnostic)]
pub(crate) enum UnescapeError {
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
            "escape the character",
            applicability = "machine-applicable",
            code = "{escaped_sugg}",
            style = "verbose"
        )]
        char_span: Span,
        escaped_sugg: String,
        escaped_msg: String,
        byte: bool,
    },
    #[diag(
        r#"{$double_quotes ->
            [true] bare CR not allowed in string, use `\r` instead
            *[false] character constant must be escaped: `\r`
        }"#
    )]
    BareCr {
        #[primary_span]
        #[suggestion(
            "escape the character",
            applicability = "machine-applicable",
            code = "\\r",
            style = "verbose"
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
        #[label(r#"missing a closing `{"}"}`"#)]
        Span,
        #[suggestion(
            "terminate the unicode escape",
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
    #[diag("null characters in C string literals are not supported")]
    NulInCStr {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum MoreThanOneCharSugg {
    #[suggestion(
        "consider using the normalized form `{$ch}` of this character",
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
        "consider removing the non-printing characters",
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
        "if you meant to write a {$is_byte ->
            [true] byte string
            *[false] string
        } literal, use double quotes",
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
    #[multipart_suggestion(
        "if you meant to write a {$is_byte ->
            [true] byte string
            *[false] string
        } literal, use double quotes",
        applicability = "machine-applicable"
    )]
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
pub(crate) enum MoreThanOneCharNote {
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
pub(crate) enum NoBraceUnicodeSub {
    #[suggestion(
        "format of unicode escape sequences uses braces",
        code = "{suggestion}",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    Suggestion {
        #[primary_span]
        span: Span,
        suggestion: String,
    },
    #[help(r#"format of unicode escape sequences is `\u{"{...}"}`"#)]
    Help,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("wrap the pattern in parentheses", applicability = "machine-applicable")]
pub(crate) struct WrapInParens {
    #[suggestion_part(code = "(")]
    pub(crate) lo: Span,
    #[suggestion_part(code = ")")]
    pub(crate) hi: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum TopLevelOrPatternNotAllowedSugg {
    #[suggestion(
        "remove the `|`",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
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
#[diag("unexpected `||` before function parameter")]
#[note("alternatives in or-patterns are separated with `|`, not `||`")]
pub(crate) struct UnexpectedVertVertBeforeFunctionParam {
    #[primary_span]
    #[suggestion(
        "remove the `||`",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected token `||` in pattern")]
pub(crate) struct UnexpectedVertVertInPattern {
    #[primary_span]
    #[suggestion(
        "use a single `|` to separate multiple alternative patterns",
        code = "|",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
    #[label("while parsing this or-pattern starting here")]
    pub start: Option<Span>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "a trailing `{$token}` is not allowed in an or-pattern",
    code = "",
    applicability = "machine-applicable",
    style = "tool-only"
)]
pub(crate) struct TrailingVertSuggestion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("a trailing `{$token}` is not allowed in an or-pattern")]
pub(crate) struct TrailingVertNotAllowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: TrailingVertSuggestion,
    #[label("while parsing this or-pattern starting here")]
    pub start: Option<Span>,
    pub token: Token,
    #[note("alternatives in or-patterns are separated with `|`, not `||`")]
    pub note_double_vert: bool,
}

#[derive(Diagnostic)]
#[diag("unexpected `...`")]
pub(crate) struct DotDotDotRestPattern {
    #[primary_span]
    #[label("not a valid pattern")]
    pub span: Span,
    #[suggestion(
        "for a rest pattern, use `..` instead of `...`",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub suggestion: Option<Span>,
    #[note(
        "only `extern \"C\"` and `extern \"C-unwind\"` functions may have a C variable argument list"
    )]
    pub var_args: Option<()>,
}

#[derive(Diagnostic)]
#[diag("pattern on wrong side of `@`")]
pub(crate) struct PatternOnWrongSideOfAt {
    #[primary_span]
    #[suggestion(
        "switch the order",
        code = "{whole_pat}",
        applicability = "machine-applicable",
        style = "verbose"
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

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "add parentheses to clarify the precedence",
    applicability = "machine-applicable"
)]
pub(crate) struct ParenRangeSuggestion {
    #[suggestion_part(code = "(")]
    pub lo: Span,
    #[suggestion_part(code = ")")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("the range pattern here has ambiguous interpretation")]
pub(crate) struct AmbiguousRangePattern {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: ParenRangeSuggestion,
}

#[derive(Diagnostic)]
#[diag("unexpected lifetime `{$symbol}` in pattern")]
pub(crate) struct UnexpectedLifetimeInPattern {
    #[primary_span]
    pub span: Span,
    pub symbol: Symbol,
    #[suggestion(
        "remove the lifetime",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
pub(crate) enum InvalidMutInPattern {
    #[diag("`mut` must be attached to each individual binding")]
    #[note("`mut` may be followed by `variable` and `variable @ pattern`")]
    NestedIdent {
        #[primary_span]
        #[suggestion(
            "add `mut` to each binding",
            code = "{pat}",
            applicability = "machine-applicable",
            style = "verbose"
        )]
        span: Span,
        pat: String,
    },
    #[diag("`mut` must be followed by a named binding")]
    #[note("`mut` may be followed by `variable` and `variable @ pattern`")]
    NonIdent {
        #[primary_span]
        #[suggestion(
            "remove the `mut` prefix",
            code = "",
            applicability = "machine-applicable",
            style = "verbose"
        )]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("`mut` on a binding may not be repeated")]
pub(crate) struct RepeatedMutInPattern {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove the additional `mut`s",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("range-to patterns with `...` are not allowed")]
pub(crate) struct DotDotDotRangeToPatternNotAllowed {
    #[primary_span]
    #[suggestion(
        "use `..=` instead",
        style = "verbose",
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
#[diag("`@ ..` is not supported in struct patterns")]
pub(crate) struct AtDotDotInStructPattern {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "bind to each field separately or, if you don't need them, just remove `{$ident} @`",
        code = "",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    pub remove: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag("unexpected `@` in struct pattern")]
#[note("struct patterns use `field: pattern` syntax to bind to fields")]
#[help(
    "consider replacing `new_name @ field_name` with `field_name: new_name` if that is what you intended"
)]
pub(crate) struct AtInStructPattern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected field pattern, found `{$token_str}`")]
pub(crate) struct DotDotDotForRemainingFields {
    #[primary_span]
    #[suggestion(
        "to omit remaining fields, use `..`",
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
#[diag(
    "expected {$is_bound ->
        [true] a pattern range bound
        *[false] a pattern
    }, found an expression"
)]
#[note(
    "arbitrary expressions are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>"
)]
pub(crate) struct UnexpectedExpressionInPattern {
    /// The unexpected expr's span.
    #[primary_span]
    #[label("not a pattern")]
    pub span: Span,
    /// Was a `RangePatternBound` expected?
    pub is_bound: bool,
    /// The unexpected expr's precedence (used in match arm guard suggestions).
    pub expr_precedence: ExprPrecedence,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnexpectedExpressionInPatternSugg {
    #[multipart_suggestion(
        "consider moving the expression to a match arm guard",
        applicability = "maybe-incorrect"
    )]
    CreateGuard {
        /// Where to put the suggested identifier.
        #[suggestion_part(code = "{ident}")]
        ident_span: Span,
        /// Where to put the match arm.
        #[suggestion_part(code = " if {ident} == {expr}")]
        pat_hi: Span,
        /// The suggested identifier.
        ident: String,
        /// The unexpected expression.
        expr: String,
    },

    #[multipart_suggestion(
        "consider moving the expression to the match arm guard",
        applicability = "maybe-incorrect"
    )]
    UpdateGuard {
        /// Where to put the suggested identifier.
        #[suggestion_part(code = "{ident}")]
        ident_span: Span,
        /// The beginning of the match arm guard's expression (insert a `(` if `Some`).
        #[suggestion_part(code = "(")]
        guard_lo: Option<Span>,
        /// The end of the match arm guard's expression.
        #[suggestion_part(code = "{guard_hi_paren} && {ident} == {expr}")]
        guard_hi: Span,
        /// Either `")"` or `""`.
        guard_hi_paren: &'static str,
        /// The suggested identifier.
        ident: String,
        /// The unexpected expression.
        expr: String,
    },

    #[multipart_suggestion(
        "consider extracting the expression into a `const`",
        applicability = "has-placeholders"
    )]
    Const {
        /// Where to put the extracted constant declaration.
        #[suggestion_part(code = "{indentation}const {ident}: /* Type */ = {expr};\n")]
        stmt_lo: Span,
        /// Where to put the suggested identifier.
        #[suggestion_part(code = "{ident}")]
        ident_span: Span,
        /// The suggested identifier.
        ident: String,
        /// The unexpected expression.
        expr: String,
        /// The statement's block's indentation.
        indentation: String,
    },
}

#[derive(Diagnostic)]
#[diag("range pattern bounds cannot have parentheses")]
pub(crate) struct UnexpectedParenInRangePat {
    #[primary_span]
    pub span: Vec<Span>,
    #[subdiagnostic]
    pub sugg: UnexpectedParenInRangePatSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("remove these parentheses", applicability = "machine-applicable")]
pub(crate) struct UnexpectedParenInRangePatSugg {
    #[suggestion_part(code = "")]
    pub start_span: Span,
    #[suggestion_part(code = "")]
    pub end_span: Span,
}

#[derive(Diagnostic)]
#[diag("return types are denoted using `->`")]
pub(crate) struct ReturnTypesUseThinArrow {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use `->` instead",
        style = "verbose",
        code = " -> ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("lifetimes must be followed by `+` to form a trait object type")]
pub(crate) struct NeedPlusAfterTraitObjectLifetime {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "consider adding a trait bound after the potential lifetime bound",
        code = " + /* Trait */",
        applicability = "has-placeholders"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("expected `mut` or `const` keyword in raw pointer type")]
pub(crate) struct ExpectedMutOrConstInRawPointerType {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "add `mut` or `const` here",
        code("mut ", "const "),
        applicability = "has-placeholders",
        style = "verbose"
    )]
    pub after_asterisk: Span,
}

#[derive(Diagnostic)]
#[diag("lifetime must precede `mut`")]
pub(crate) struct LifetimeAfterMut {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "place the lifetime before `mut`",
        code = "&{snippet} mut",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub suggest_lifetime: Option<Span>,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag("`mut` must precede `dyn`")]
pub(crate) struct DynAfterMut {
    #[primary_span]
    #[suggestion(
        "place `mut` before `dyn`",
        code = "&mut dyn",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("an `fn` pointer type cannot be `const`")]
#[note("allowed qualifiers are: `unsafe` and `extern`")]
pub(crate) struct FnPointerCannotBeConst {
    #[primary_span]
    #[label("`const` because of this")]
    pub span: Span,
    #[suggestion(
        "remove the `const` qualifier",
        code = "",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("an `fn` pointer type cannot be `async`")]
#[note("allowed qualifiers are: `unsafe` and `extern`")]
pub(crate) struct FnPointerCannotBeAsync {
    #[primary_span]
    #[label("`async` because of this")]
    pub span: Span,
    #[suggestion(
        "remove the `async` qualifier",
        code = "",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("C-variadic type `...` may not be nested inside another type", code = E0743)]
pub(crate) struct NestedCVariadicType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected `...`")]
#[note(
    "only `extern \"C\"` and `extern \"C-unwind\"` functions may have a C variable argument list"
)]
pub(crate) struct InvalidCVariadicType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid `dyn` keyword")]
#[help("`dyn` is only needed at the start of a trait `+`-separated list")]
pub(crate) struct InvalidDynKeyword {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove this keyword",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum HelpUseLatestEdition {
    #[help("set `edition = \"{$edition}\"` in `Cargo.toml`")]
    #[note("for more on editions, read https://doc.rust-lang.org/edition-guide")]
    Cargo { edition: Edition },
    #[help("pass `--edition {$edition}` to `rustc`")]
    #[note("for more on editions, read https://doc.rust-lang.org/edition-guide")]
    Standalone { edition: Edition },
}

impl HelpUseLatestEdition {
    pub(crate) fn new() -> Self {
        let edition = LATEST_STABLE_EDITION;
        if rustc_session::utils::was_invoked_from_cargo() {
            Self::Cargo { edition }
        } else {
            Self::Standalone { edition }
        }
    }
}

#[derive(Diagnostic)]
#[diag("`box_syntax` has been removed")]
pub(crate) struct BoxSyntaxRemoved {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: AddBoxNew,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use `Box::new()` instead",
    applicability = "machine-applicable",
    style = "verbose"
)]
pub(crate) struct AddBoxNew {
    #[suggestion_part(code = "Box::new(")]
    pub box_kw_and_lo: Span,
    #[suggestion_part(code = ")")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("return type not allowed with return type notation")]
pub(crate) struct BadReturnTypeNotationOutput {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "remove the return type",
        code = "",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub suggestion: Span,
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
#[diag("lifetimes are not permitted in this context")]
#[help("if you meant to specify a trait object, write `dyn /* Trait */ + {$lifetime}`")]
pub(crate) struct LifetimeInEqConstraint {
    #[primary_span]
    #[label("lifetime is not allowed here")]
    pub span: Span,
    pub lifetime: Ident,
    #[label("this introduces an associated item binding")]
    pub binding_label: Span,
    #[suggestion(
        "you might have meant to write a bound here",
        style = "verbose",
        applicability = "maybe-incorrect",
        code = ": "
    )]
    pub colon_sugg: Span,
}

#[derive(Diagnostic)]
#[diag("`{$modifier}` may only modify trait bounds, not lifetime bounds")]
pub(crate) struct ModifierLifetime {
    #[primary_span]
    #[suggestion(
        "remove the `{$modifier}`",
        style = "tool-only",
        applicability = "maybe-incorrect",
        code = ""
    )]
    pub span: Span,
    pub modifier: &'static str,
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
        "labels start with a tick",
        code = "'",
        applicability = "machine-applicable",
        style = "verbose"
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
        "items are imported using the `use` keyword",
        code = "use",
        applicability = "machine-applicable",
        style = "verbose"
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
        "use double colon",
        code = "::",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$descr} is not supported in {$ctx}")]
pub(crate) struct BadItemKind {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
    pub ctx: &'static str,
    #[help("consider moving the {$descr} out to a nearby module scope")]
    pub help: bool,
}

#[derive(Diagnostic)]
#[diag("expected `!` after `macro_rules`")]
pub(crate) struct MacroRulesMissingBang {
    #[primary_span]
    pub span: Span,
    #[suggestion("add a `!`", code = "!", applicability = "machine-applicable", style = "verbose")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("macro names aren't followed by a `!`")]
pub(crate) struct MacroNameRemoveBang {
    #[primary_span]
    #[suggestion(
        "remove the `!`",
        code = "",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("can't qualify macro_rules invocation with `{$vis}`")]
pub(crate) struct MacroRulesVisibility<'a> {
    #[primary_span]
    #[suggestion(
        "try exporting the macro",
        code = "#[macro_export]",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub span: Span,
    pub vis: &'a str,
}

#[derive(Diagnostic)]
#[diag("can't qualify macro invocation with `pub`")]
#[help("try adjusting the macro to put `{$vis}` inside the invocation")]
pub(crate) struct MacroInvocationVisibility<'a> {
    #[primary_span]
    #[suggestion(
        "remove the visibility",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
    pub vis: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$kw_str}` definition cannot be nested inside `{$keyword}`")]
pub(crate) struct NestedAdt<'a> {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "consider creating a new `{$kw_str}` definition instead of nesting",
        code = "",
        applicability = "maybe-incorrect",
        style = "verbose"
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
    r#"surround the expression with `{"{"}` and `{"}"}` instead of `=` and `;`"#,
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
        "escape `box` to use it as an identifier",
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
        "remove extra angle {$plural ->
            [true] brackets
            *[false] bracket
        }",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub span: Span,
    pub plural: bool,
}

#[derive(Diagnostic)]
#[diag("expected `+` between lifetime and {$sym}")]
pub(crate) struct MissingPlusBounds {
    #[primary_span]
    pub span: Span,
    #[suggestion("add `+`", code = " +", applicability = "maybe-incorrect", style = "verbose")]
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
#[multipart_suggestion("fix the parentheses", applicability = "machine-applicable")]
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
        "write it in {$case}",
        code = "{kw}",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub kw: &'a str,
    pub case: Case,
}

pub(crate) enum Case {
    Upper,
    Lower,
    Mixed,
}

impl IntoDiagArg for Case {
    fn into_diag_arg(self, path: &mut Option<PathBuf>) -> DiagArgValue {
        match self {
            Case::Upper => "uppercase",
            Case::Lower => "lowercase",
            Case::Mixed => "the correct case",
        }
        .into_diag_arg(path)
    }
}

#[derive(Diagnostic)]
#[diag("unknown `builtin #` construct `{$name}`")]
pub(crate) struct UnknownBuiltinConstruct {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
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
    "move the body before the where clause",
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
        "use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments",
        style = "verbose",
        code = "::",
        applicability = "maybe-incorrect"
    )]
    pub suggest_turbofish: Span,
}

#[derive(Diagnostic)]
#[diag("`for<...>` expected after `{$kw}`, not before")]
pub(crate) struct TransposeDynOrImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub kw: &'a str,
    #[subdiagnostic]
    pub sugg: TransposeDynOrImplSugg<'a>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("move `{$kw}` before the `for<...>`", applicability = "machine-applicable")]
pub(crate) struct TransposeDynOrImplSugg<'a> {
    #[suggestion_part(code = "")]
    pub removal_span: Span,
    #[suggestion_part(code = "{kw} ")]
    pub insertion_span: Span,
    pub kw: &'a str,
}

#[derive(Diagnostic)]
#[diag("array indexing not supported in offset_of")]
pub(crate) struct ArrayIndexInOffsetOf(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag("offset_of expects dot-separated field and variant names")]
pub(crate) struct InvalidOffsetOf(#[primary_span] pub Span);

#[derive(Diagnostic)]
#[diag("`async` trait implementations are unsupported")]
pub(crate) struct AsyncImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`->` is not valid syntax for field accesses and method calls")]
#[help(
    "the `.` operator will automatically dereference the value, except if the value is a raw pointer"
)]
pub(crate) struct ExprRArrowCall {
    #[primary_span]
    #[suggestion(
        "try using `.` instead",
        style = "verbose",
        applicability = "machine-applicable",
        code = "."
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attributes are not allowed on range expressions starting with `..`")]
pub(crate) struct DotDotRangeAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`for<...>` binder should be placed before trait bound modifiers")]
pub(crate) struct BinderBeforeModifiers {
    #[primary_span]
    pub binder_span: Span,
    #[label("place the `for<...>` binder before any modifiers")]
    pub modifiers_span: Span,
}

#[derive(Diagnostic)]
#[diag("`for<...>` binder not allowed with `{$polarity}` trait polarity modifier")]
pub(crate) struct BinderAndPolarity {
    #[primary_span]
    pub polarity_span: Span,
    #[label("there is not a well-defined meaning for a higher-ranked `{$polarity}` trait")]
    pub binder_span: Span,
    pub polarity: &'static str,
}

#[derive(Diagnostic)]
#[diag("`{$modifiers_concatenated}` trait not allowed with `{$polarity}` trait polarity modifier")]
pub(crate) struct PolarityAndModifiers {
    #[primary_span]
    pub polarity_span: Span,
    #[label(
        "there is not a well-defined meaning for a `{$modifiers_concatenated} {$polarity}` trait"
    )]
    pub modifiers_span: Span,
    pub polarity: &'static str,
    pub modifiers_concatenated: String,
}

#[derive(Diagnostic)]
#[diag("type not allowed for shorthand `self` parameter")]
pub(crate) struct IncorrectTypeOnSelf {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub move_self_modifier: MoveSelfModifier,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "move the modifiers on `self` to the type",
    applicability = "machine-applicable"
)]
pub(crate) struct MoveSelfModifier {
    #[suggestion_part(code = "")]
    pub removal_span: Span,
    #[suggestion_part(code = "{modifier}")]
    pub insertion_span: Span,
    pub modifier: String,
}

#[derive(Diagnostic)]
#[diag("the `{$symbol}` operand cannot be used with `{$macro_name}!`")]
pub(crate) struct AsmUnsupportedOperand<'a> {
    #[primary_span]
    #[label(
        "the `{$symbol}` operand is not meaningful for global-scoped inline assembly, remove it"
    )]
    pub(crate) span: Span,
    pub(crate) symbol: &'a str,
    pub(crate) macro_name: &'static str,
}

#[derive(Diagnostic)]
#[diag("_ cannot be used for input operands")]
pub(crate) struct AsmUnderscoreInput {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a path for argument to `sym`")]
pub(crate) struct AsmSymNoPath {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("requires at least a template string argument")]
pub(crate) struct AsmRequiresTemplate {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected token: `,`")]
pub(crate) struct AsmExpectedComma {
    #[primary_span]
    #[label("expected `,`")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "expected operand, {$is_inline_asm ->
        [false] options
        *[true] clobber_abi, options
    }, or additional template string"
)]
pub(crate) struct AsmExpectedOther {
    #[primary_span]
    #[label(
        "expected operand, {$is_inline_asm ->
            [false] options
            *[true] clobber_abi, options
        }, or additional template string"
    )]
    pub(crate) span: Span,
    pub(crate) is_inline_asm: bool,
}

#[derive(Diagnostic)]
#[diag("at least one abi must be provided as an argument to `clobber_abi`")]
pub(crate) struct NonABI {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected string literal")]
pub(crate) struct AsmExpectedStringLiteral {
    #[primary_span]
    #[label("not a string literal")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected register class or explicit register")]
pub(crate) struct ExpectedRegisterClassOrExplicitRegister {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(LintDiagnostic)]
#[diag("unicode codepoint changing visible direction of text present in {$label}")]
#[note(
    "these kind of unicode codepoints change the way text flows on applications that support them, but can cause confusion because they change the order of characters on the screen"
)]
pub(crate) struct HiddenUnicodeCodepointsDiag {
    pub label: String,
    pub count: usize,
    #[label(
        "this {$label} contains {$count ->
            [one] an invisible
            *[other] invisible
        } unicode text flow control {$count ->
            [one] codepoint
            *[other] codepoints
        }"
    )]
    pub span_label: Span,
    #[subdiagnostic]
    pub labels: Option<HiddenUnicodeCodepointsDiagLabels>,
    #[subdiagnostic]
    pub sub: HiddenUnicodeCodepointsDiagSub,
}

pub(crate) struct HiddenUnicodeCodepointsDiagLabels {
    pub spans: Vec<(char, Span)>,
}

impl Subdiagnostic for HiddenUnicodeCodepointsDiagLabels {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        for (c, span) in self.spans {
            diag.span_label(span, format!("{c:?}"));
        }
    }
}

pub(crate) enum HiddenUnicodeCodepointsDiagSub {
    Escape { spans: Vec<(char, Span)> },
    NoEscape { spans: Vec<(char, Span)> },
}

// Used because of multiple multipart_suggestion and note
impl Subdiagnostic for HiddenUnicodeCodepointsDiagSub {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            HiddenUnicodeCodepointsDiagSub::Escape { spans } => {
                diag.multipart_suggestion_with_style(
                    inline_fluent!("if their presence wasn't intentional, you can remove them"),
                    spans.iter().map(|(_, span)| (*span, "".to_string())).collect(),
                    Applicability::MachineApplicable,
                    SuggestionStyle::HideCodeAlways,
                );
                diag.multipart_suggestion(
                    inline_fluent!("if you want to keep them but make them visible in your source code, you can escape them"),
                    spans
                        .into_iter()
                        .map(|(c, span)| {
                            let c = format!("{c:?}");
                            (span, c[1..c.len() - 1].to_string())
                        })
                        .collect(),
                    Applicability::MachineApplicable,
                );
            }
            HiddenUnicodeCodepointsDiagSub::NoEscape { spans } => {
                // FIXME: in other suggestions we've reversed the inner spans of doc comments. We
                // should do the same here to provide the same good suggestions as we do for
                // literals above.
                diag.arg(
                    "escaped",
                    spans
                        .into_iter()
                        .map(|(c, _)| format!("{c:?}"))
                        .collect::<Vec<String>>()
                        .join(", "),
                );
                diag.note(inline_fluent!(
                    "if their presence wasn't intentional, you can remove them"
                ));
                diag.note(inline_fluent!("if you want to keep them but make them visible in your source code, you can escape them: {$escaped}"));
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag("missing pattern for `...` argument")]
pub(crate) struct VarargsWithoutPattern {
    #[suggestion(
        "name the argument, or use `_` to continue ignoring it",
        code = "_: ...",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("only trait impls can be reused")]
pub(crate) struct ImplReuseInherentImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("placeholder `_` is not allowed for the path in struct literals")]
pub(crate) struct StructLiteralPlaceholderPath {
    #[primary_span]
    #[label("not allowed in struct literals")]
    #[suggestion(
        "replace it with the correct type",
        applicability = "has-placeholders",
        code = "/* Type */",
        style = "verbose"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("struct literal body without path")]
pub(crate) struct StructLiteralWithoutPathLate {
    #[primary_span]
    #[label("struct name missing for struct literal")]
    pub span: Span,
    #[suggestion(
        "add the correct type",
        applicability = "has-placeholders",
        code = "/* Type */ ",
        style = "verbose"
    )]
    pub suggestion_span: Span,
}

/// Used to forbid `let` expressions in certain syntactic locations.
#[derive(Clone, Copy, Subdiagnostic)]
pub(crate) enum ForbiddenLetReason {
    /// `let` is not valid and the source environment is not important
    OtherForbidden,
    /// A let chain with the `||` operator
    #[note("`||` operators are not supported in let chain expressions")]
    NotSupportedOr(#[primary_span] Span),
    /// A let chain with invalid parentheses
    ///
    /// For example, `let 1 = 1 && (expr && expr)` is allowed
    /// but `(let 1 = 1 && (let 1 = 1 && (let 1 = 1))) && let a = 1` is not
    #[note("`let`s wrapped in parentheses are not supported in a context with let chains")]
    NotSupportedParentheses(#[primary_span] Span),
}

#[derive(Debug, rustc_macros::Subdiagnostic)]
#[suggestion(
    "{$is_incorrect_case ->
        [true] write keyword `{$similar_kw}` in lowercase
        *[false] there is a keyword `{$similar_kw}` with a similar name
    }",
    applicability = "machine-applicable",
    code = "{similar_kw}",
    style = "verbose"
)]
pub(crate) struct MisspelledKw {
    // We use a String here because `Symbol::into_diag_arg` calls `Symbol::to_ident_string`, which
    // prefix the keyword with a `r#` because it aims to print the symbol as an identifier.
    pub similar_kw: String,
    #[primary_span]
    pub span: Span,
    pub is_incorrect_case: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TokenDescription {
    ReservedIdentifier,
    Keyword,
    ReservedKeyword,
    DocComment,

    // Expanded metavariables are wrapped in invisible delimiters which aren't
    // pretty-printed. In error messages we must handle these specially
    // otherwise we get confusing things in messages like "expected `(`, found
    // ``". It's better to say e.g. "expected `(`, found type metavariable".
    MetaVar(MetaVarKind),
}

impl TokenDescription {
    pub(super) fn from_token(token: &Token) -> Option<Self> {
        match token.kind {
            _ if token.is_special_ident() => Some(TokenDescription::ReservedIdentifier),
            _ if token.is_used_keyword() => Some(TokenDescription::Keyword),
            _ if token.is_unused_keyword() => Some(TokenDescription::ReservedKeyword),
            token::DocComment(..) => Some(TokenDescription::DocComment),
            token::OpenInvisible(InvisibleOrigin::MetaVar(kind)) => {
                Some(TokenDescription::MetaVar(kind))
            }
            _ => None,
        }
    }
}
