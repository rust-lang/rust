use std::mem::take;
use std::ops::{Deref, DerefMut};

use ast::token::IdentIsRaw;
use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Lit, LitKind, Token, TokenKind};
use rustc_ast::util::parser::AssocOp;
use rustc_ast::{
    AngleBracketedArg, AngleBracketedArgs, AnonConst, AttrVec, BinOpKind, BindingMode, Block,
    BlockCheckMode, Expr, ExprKind, GenericArg, Generics, Item, ItemKind, Param, Pat, PatKind,
    Path, PathSegment, QSelf, Recovered, Ty, TyKind,
};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, ErrorGuaranteed, PResult, Subdiagnostic, Suggestions,
    pluralize,
};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::used_keywords;
use rustc_span::{BytePos, DUMMY_SP, Ident, Span, SpanSnippetError, Symbol, kw, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::{debug, trace};

use super::pat::Expected;
use super::{
    BlockMode, CommaRecoveryMode, ExpTokenPair, Parser, PathStyle, Restrictions, SemiColonMode,
    SeqSep, TokenType,
};
use crate::errors::{
    AddParen, AmbiguousPlus, AsyncMoveBlockIn2015, AsyncUseBlockIn2015, AttributeOnParamType,
    AwaitSuggestion, BadQPathStage2, BadTypePlus, BadTypePlusSub, ColonAsSemi,
    ComparisonOperatorsCannotBeChained, ComparisonOperatorsCannotBeChainedSugg,
    ConstGenericWithoutBraces, ConstGenericWithoutBracesSugg, DocCommentDoesNotDocumentAnything,
    DocCommentOnParamType, DoubleColonInBound, ExpectedIdentifier, ExpectedSemi, ExpectedSemiSugg,
    GenericParamsWithoutAngleBrackets, GenericParamsWithoutAngleBracketsSugg,
    HelpIdentifierStartsWithNumber, HelpUseLatestEdition, InInTypo, IncorrectAwait,
    IncorrectSemicolon, IncorrectUseOfAwait, IncorrectUseOfUse, PatternMethodParamWithoutBody,
    QuestionMarkInType, QuestionMarkInTypeSugg, SelfParamNotFirst, StructLiteralBodyWithoutPath,
    StructLiteralBodyWithoutPathSugg, SuggAddMissingLetStmt, SuggEscapeIdentifier, SuggRemoveComma,
    TernaryOperator, TernaryOperatorSuggestion, UnexpectedConstInGenericParam,
    UnexpectedConstParamDeclaration, UnexpectedConstParamDeclarationSugg, UnmatchedAngleBrackets,
    UseEqInstead, WrapType,
};
use crate::parser::attr::InnerAttrPolicy;
use crate::{exp, fluent_generated as fluent};

/// Creates a placeholder argument.
pub(super) fn dummy_arg(ident: Ident, guar: ErrorGuaranteed) -> Param {
    let pat = P(Pat {
        id: ast::DUMMY_NODE_ID,
        kind: PatKind::Ident(BindingMode::NONE, ident, None),
        span: ident.span,
        tokens: None,
    });
    let ty = Ty { kind: TyKind::Err(guar), span: ident.span, id: ast::DUMMY_NODE_ID, tokens: None };
    Param {
        attrs: AttrVec::default(),
        id: ast::DUMMY_NODE_ID,
        pat,
        span: ident.span,
        ty: P(ty),
        is_placeholder: false,
    }
}

pub(super) trait RecoverQPath: Sized + 'static {
    const PATH_STYLE: PathStyle = PathStyle::Expr;
    fn to_ty(&self) -> Option<P<Ty>>;
    fn recovered(qself: Option<P<QSelf>>, path: ast::Path) -> Self;
}

impl RecoverQPath for Ty {
    const PATH_STYLE: PathStyle = PathStyle::Type;
    fn to_ty(&self) -> Option<P<Ty>> {
        Some(P(self.clone()))
    }
    fn recovered(qself: Option<P<QSelf>>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            kind: TyKind::Path(qself, path),
            id: ast::DUMMY_NODE_ID,
            tokens: None,
        }
    }
}

impl RecoverQPath for Pat {
    const PATH_STYLE: PathStyle = PathStyle::Pat;
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn recovered(qself: Option<P<QSelf>>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            kind: PatKind::Path(qself, path),
            id: ast::DUMMY_NODE_ID,
            tokens: None,
        }
    }
}

impl RecoverQPath for Expr {
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn recovered(qself: Option<P<QSelf>>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            kind: ExprKind::Path(qself, path),
            attrs: AttrVec::new(),
            id: ast::DUMMY_NODE_ID,
            tokens: None,
        }
    }
}

/// Control whether the closing delimiter should be consumed when calling `Parser::consume_block`.
pub(crate) enum ConsumeClosingDelim {
    Yes,
    No,
}

#[derive(Clone, Copy)]
pub enum AttemptLocalParseRecovery {
    Yes,
    No,
}

impl AttemptLocalParseRecovery {
    pub(super) fn yes(&self) -> bool {
        match self {
            AttemptLocalParseRecovery::Yes => true,
            AttemptLocalParseRecovery::No => false,
        }
    }

    pub(super) fn no(&self) -> bool {
        match self {
            AttemptLocalParseRecovery::Yes => false,
            AttemptLocalParseRecovery::No => true,
        }
    }
}

/// Information for emitting suggestions and recovering from
/// C-style `i++`, `--i`, etc.
#[derive(Debug, Copy, Clone)]
struct IncDecRecovery {
    /// Is this increment/decrement its own statement?
    standalone: IsStandalone,
    /// Is this an increment or decrement?
    op: IncOrDec,
    /// Is this pre- or postfix?
    fixity: UnaryFixity,
}

/// Is an increment or decrement expression its own statement?
#[derive(Debug, Copy, Clone)]
enum IsStandalone {
    /// It's standalone, i.e., its own statement.
    Standalone,
    /// It's a subexpression, i.e., *not* standalone.
    Subexpr,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum IncOrDec {
    Inc,
    Dec,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum UnaryFixity {
    Pre,
    Post,
}

impl IncOrDec {
    fn chr(&self) -> char {
        match self {
            Self::Inc => '+',
            Self::Dec => '-',
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Inc => "increment",
            Self::Dec => "decrement",
        }
    }
}

impl std::fmt::Display for UnaryFixity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pre => write!(f, "prefix"),
            Self::Post => write!(f, "postfix"),
        }
    }
}

#[derive(Debug, rustc_macros::Subdiagnostic)]
#[suggestion(
    parse_misspelled_kw,
    applicability = "machine-applicable",
    code = "{similar_kw}",
    style = "verbose"
)]
struct MisspelledKw {
    similar_kw: String,
    #[primary_span]
    span: Span,
    is_incorrect_case: bool,
}

/// Checks if the given `lookup` identifier is similar to any keyword symbol in `candidates`.
fn find_similar_kw(lookup: Ident, candidates: &[Symbol]) -> Option<MisspelledKw> {
    let lowercase = lookup.name.as_str().to_lowercase();
    let lowercase_sym = Symbol::intern(&lowercase);
    if candidates.contains(&lowercase_sym) {
        Some(MisspelledKw { similar_kw: lowercase, span: lookup.span, is_incorrect_case: true })
    } else if let Some(similar_sym) = find_best_match_for_name(candidates, lookup.name, None) {
        Some(MisspelledKw {
            similar_kw: similar_sym.to_string(),
            span: lookup.span,
            is_incorrect_case: false,
        })
    } else {
        None
    }
}

struct MultiSugg {
    msg: String,
    patches: Vec<(Span, String)>,
    applicability: Applicability,
}

impl MultiSugg {
    fn emit(self, err: &mut Diag<'_>) {
        err.multipart_suggestion(self.msg, self.patches, self.applicability);
    }

    fn emit_verbose(self, err: &mut Diag<'_>) {
        err.multipart_suggestion_verbose(self.msg, self.patches, self.applicability);
    }
}

/// SnapshotParser is used to create a snapshot of the parser
/// without causing duplicate errors being emitted when the `Parser`
/// is dropped.
pub struct SnapshotParser<'a> {
    parser: Parser<'a>,
}

impl<'a> Deref for SnapshotParser<'a> {
    type Target = Parser<'a>;

    fn deref(&self) -> &Self::Target {
        &self.parser
    }
}

impl<'a> DerefMut for SnapshotParser<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.parser
    }
}

impl<'a> Parser<'a> {
    pub fn dcx(&self) -> DiagCtxtHandle<'a> {
        self.psess.dcx()
    }

    /// Replace `self` with `snapshot.parser`.
    pub(super) fn restore_snapshot(&mut self, snapshot: SnapshotParser<'a>) {
        *self = snapshot.parser;
    }

    /// Create a snapshot of the `Parser`.
    pub fn create_snapshot_for_diagnostic(&self) -> SnapshotParser<'a> {
        let snapshot = self.clone();
        SnapshotParser { parser: snapshot }
    }

    pub(super) fn span_to_snippet(&self, span: Span) -> Result<String, SpanSnippetError> {
        self.psess.source_map().span_to_snippet(span)
    }

    /// Emits an error with suggestions if an identifier was expected but not found.
    ///
    /// Returns a possibly recovered identifier.
    pub(super) fn expected_ident_found(
        &mut self,
        recover: bool,
    ) -> PResult<'a, (Ident, IdentIsRaw)> {
        let valid_follow = &[
            TokenKind::Eq,
            TokenKind::Colon,
            TokenKind::Comma,
            TokenKind::Semi,
            TokenKind::PathSep,
            TokenKind::OpenBrace,
            TokenKind::OpenParen,
            TokenKind::CloseBrace,
            TokenKind::CloseParen,
        ];
        if let TokenKind::DocComment(..) = self.prev_token.kind
            && valid_follow.contains(&self.token.kind)
        {
            let err = self.dcx().create_err(DocCommentDoesNotDocumentAnything {
                span: self.prev_token.span,
                missing_comma: None,
            });
            return Err(err);
        }

        let mut recovered_ident = None;
        // we take this here so that the correct original token is retained in
        // the diagnostic, regardless of eager recovery.
        let bad_token = self.token;

        // suggest prepending a keyword in identifier position with `r#`
        let suggest_raw = if let Some((ident, IdentIsRaw::No)) = self.token.ident()
            && ident.is_raw_guess()
            && self.look_ahead(1, |t| valid_follow.contains(&t.kind))
        {
            recovered_ident = Some((ident, IdentIsRaw::Yes));

            // `Symbol::to_string()` is different from `Symbol::into_diag_arg()`,
            // which uses `Symbol::to_ident_string()` and "helpfully" adds an implicit `r#`
            let ident_name = ident.name.to_string();

            Some(SuggEscapeIdentifier { span: ident.span.shrink_to_lo(), ident_name })
        } else {
            None
        };

        let suggest_remove_comma =
            if self.token == token::Comma && self.look_ahead(1, |t| t.is_ident()) {
                if recover {
                    self.bump();
                    recovered_ident = self.ident_or_err(false).ok();
                };

                Some(SuggRemoveComma { span: bad_token.span })
            } else {
                None
            };

        let help_cannot_start_number = self.is_lit_bad_ident().map(|(len, valid_portion)| {
            let (invalid, valid) = self.token.span.split_at(len as u32);

            recovered_ident = Some((Ident::new(valid_portion, valid), IdentIsRaw::No));

            HelpIdentifierStartsWithNumber { num_span: invalid }
        });

        let err = ExpectedIdentifier {
            span: bad_token.span,
            token: bad_token,
            suggest_raw,
            suggest_remove_comma,
            help_cannot_start_number,
        };
        let mut err = self.dcx().create_err(err);

        // if the token we have is a `<`
        // it *might* be a misplaced generic
        // FIXME: could we recover with this?
        if self.token == token::Lt {
            // all keywords that could have generic applied
            let valid_prev_keywords =
                [kw::Fn, kw::Type, kw::Struct, kw::Enum, kw::Union, kw::Trait];

            // If we've expected an identifier,
            // and the current token is a '<'
            // if the previous token is a valid keyword
            // that might use a generic, then suggest a correct
            // generic placement (later on)
            let maybe_keyword = self.prev_token;
            if valid_prev_keywords.into_iter().any(|x| maybe_keyword.is_keyword(x)) {
                // if we have a valid keyword, attempt to parse generics
                // also obtain the keywords symbol
                match self.parse_generics() {
                    Ok(generic) => {
                        if let TokenKind::Ident(symbol, _) = maybe_keyword.kind {
                            let ident_name = symbol;
                            // at this point, we've found something like
                            // `fn <T>id`
                            // and current token should be Ident with the item name (i.e. the function name)
                            // if there is a `<` after the fn name, then don't show a suggestion, show help

                            if !self.look_ahead(1, |t| *t == token::Lt)
                                && let Ok(snippet) =
                                    self.psess.source_map().span_to_snippet(generic.span)
                            {
                                err.multipart_suggestion_verbose(
                                        format!("place the generic parameter name after the {ident_name} name"),
                                        vec![
                                            (self.token.span.shrink_to_hi(), snippet),
                                            (generic.span, String::new())
                                        ],
                                        Applicability::MaybeIncorrect,
                                    );
                            } else {
                                err.help(format!(
                                    "place the generic parameter name after the {ident_name} name"
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        // if there's an error parsing the generics,
                        // then don't do a misplaced generics suggestion
                        // and emit the expected ident error instead;
                        err.cancel();
                    }
                }
            }
        }

        if let Some(recovered_ident) = recovered_ident
            && recover
        {
            err.emit();
            Ok(recovered_ident)
        } else {
            Err(err)
        }
    }

    pub(super) fn expected_ident_found_err(&mut self) -> Diag<'a> {
        self.expected_ident_found(false).unwrap_err()
    }

    /// Checks if the current token is a integer or float literal and looks like
    /// it could be a invalid identifier with digits at the start.
    ///
    /// Returns the number of characters (bytes) composing the invalid portion
    /// of the identifier and the valid portion of the identifier.
    pub(super) fn is_lit_bad_ident(&mut self) -> Option<(usize, Symbol)> {
        // ensure that the integer literal is followed by a *invalid*
        // suffix: this is how we know that it is a identifier with an
        // invalid beginning.
        if let token::Literal(Lit {
            kind: token::LitKind::Integer | token::LitKind::Float,
            symbol,
            suffix: Some(suffix), // no suffix makes it a valid literal
        }) = self.token.kind
            && rustc_ast::MetaItemLit::from_token(&self.token).is_none()
        {
            Some((symbol.as_str().len(), suffix))
        } else {
            None
        }
    }

    pub(super) fn expected_one_of_not_found(
        &mut self,
        edible: &[ExpTokenPair<'_>],
        inedible: &[ExpTokenPair<'_>],
    ) -> PResult<'a, ErrorGuaranteed> {
        debug!("expected_one_of_not_found(edible: {:?}, inedible: {:?})", edible, inedible);
        fn tokens_to_string(tokens: &[TokenType]) -> String {
            let mut i = tokens.iter();
            // This might be a sign we need a connect method on `Iterator`.
            let b = i.next().map_or_else(String::new, |t| t.to_string());
            i.enumerate().fold(b, |mut b, (i, a)| {
                if tokens.len() > 2 && i == tokens.len() - 2 {
                    b.push_str(", or ");
                } else if tokens.len() == 2 && i == tokens.len() - 2 {
                    b.push_str(" or ");
                } else {
                    b.push_str(", ");
                }
                b.push_str(&a.to_string());
                b
            })
        }

        for exp in edible.iter().chain(inedible.iter()) {
            self.expected_token_types.insert(exp.token_type);
        }
        let mut expected: Vec<_> = self.expected_token_types.iter().collect();
        expected.sort_by_cached_key(|x| x.to_string());
        expected.dedup();

        let sm = self.psess.source_map();

        // Special-case "expected `;`" errors.
        if expected.contains(&TokenType::Semi) {
            // If the user is trying to write a ternary expression, recover it and
            // return an Err to prevent a cascade of irrelevant diagnostics.
            if self.prev_token == token::Question
                && let Err(e) = self.maybe_recover_from_ternary_operator(None)
            {
                return Err(e);
            }

            if self.token.span == DUMMY_SP || self.prev_token.span == DUMMY_SP {
                // Likely inside a macro, can't provide meaningful suggestions.
            } else if !sm.is_multiline(self.prev_token.span.until(self.token.span)) {
                // The current token is in the same line as the prior token, not recoverable.
            } else if [token::Comma, token::Colon].contains(&self.token.kind)
                && self.prev_token == token::CloseParen
            {
                // Likely typo: The current token is on a new line and is expected to be
                // `.`, `;`, `?`, or an operator after a close delimiter token.
                //
                // let a = std::process::Command::new("echo")
                //         .arg("1")
                //         ,arg("2")
                //         ^
                // https://github.com/rust-lang/rust/issues/72253
            } else if self.look_ahead(1, |t| {
                t == &token::CloseBrace || t.can_begin_expr() && *t != token::Colon
            }) && [token::Comma, token::Colon].contains(&self.token.kind)
            {
                // Likely typo: `,` → `;` or `:` → `;`. This is triggered if the current token is
                // either `,` or `:`, and the next token could either start a new statement or is a
                // block close. For example:
                //
                //   let x = 32:
                //   let y = 42;
                let guar = self.dcx().emit_err(ExpectedSemi {
                    span: self.token.span,
                    token: self.token,
                    unexpected_token_label: None,
                    sugg: ExpectedSemiSugg::ChangeToSemi(self.token.span),
                });
                self.bump();
                return Ok(guar);
            } else if self.look_ahead(0, |t| {
                t == &token::CloseBrace
                    || ((t.can_begin_expr() || t.can_begin_item())
                        && t != &token::Semi
                        && t != &token::Pound)
                    // Avoid triggering with too many trailing `#` in raw string.
                    || (sm.is_multiline(
                        self.prev_token.span.shrink_to_hi().until(self.token.span.shrink_to_lo()),
                    ) && t == &token::Pound)
            }) && !expected.contains(&TokenType::Comma)
            {
                // Missing semicolon typo. This is triggered if the next token could either start a
                // new statement or is a block close. For example:
                //
                //   let x = 32
                //   let y = 42;
                let span = self.prev_token.span.shrink_to_hi();
                let guar = self.dcx().emit_err(ExpectedSemi {
                    span,
                    token: self.token,
                    unexpected_token_label: Some(self.token.span),
                    sugg: ExpectedSemiSugg::AddSemi(span),
                });
                return Ok(guar);
            }
        }

        if self.token == TokenKind::EqEq
            && self.prev_token.is_ident()
            && expected.contains(&TokenType::Eq)
        {
            // Likely typo: `=` → `==` in let expr or enum item
            return Err(self.dcx().create_err(UseEqInstead { span: self.token.span }));
        }

        if (self.token.is_keyword(kw::Move) || self.token.is_keyword(kw::Use))
            && self.prev_token.is_keyword(kw::Async)
        {
            // The 2015 edition is in use because parsing of `async move` or `async use` has failed.
            let span = self.prev_token.span.to(self.token.span);
            if self.token.is_keyword(kw::Move) {
                return Err(self.dcx().create_err(AsyncMoveBlockIn2015 { span }));
            } else {
                // kw::Use
                return Err(self.dcx().create_err(AsyncUseBlockIn2015 { span }));
            }
        }

        let expect = tokens_to_string(&expected);
        let actual = super::token_descr(&self.token);
        let (msg_exp, (label_sp, label_exp)) = if expected.len() > 1 {
            let fmt = format!("expected one of {expect}, found {actual}");
            let short_expect = if expected.len() > 6 {
                format!("{} possible tokens", expected.len())
            } else {
                expect
            };
            (fmt, (self.prev_token.span.shrink_to_hi(), format!("expected one of {short_expect}")))
        } else if expected.is_empty() {
            (
                format!("unexpected token: {actual}"),
                (self.prev_token.span, "unexpected token after this".to_string()),
            )
        } else {
            (
                format!("expected {expect}, found {actual}"),
                (self.prev_token.span.shrink_to_hi(), format!("expected {expect}")),
            )
        };
        self.last_unexpected_token_span = Some(self.token.span);
        // FIXME: translation requires list formatting (for `expect`)
        let mut err = self.dcx().struct_span_err(self.token.span, msg_exp);

        self.label_expected_raw_ref(&mut err);

        // Look for usages of '=>' where '>=' was probably intended
        if self.token == token::FatArrow
            && expected.iter().any(|tok| matches!(tok, TokenType::Operator | TokenType::Le))
            && !expected.iter().any(|tok| matches!(tok, TokenType::FatArrow | TokenType::Comma))
        {
            err.span_suggestion(
                self.token.span,
                "you might have meant to write a \"greater than or equal to\" comparison",
                ">=",
                Applicability::MaybeIncorrect,
            );
        }

        if let TokenKind::Ident(symbol, _) = &self.prev_token.kind {
            if ["def", "fun", "func", "function"].contains(&symbol.as_str()) {
                err.span_suggestion_short(
                    self.prev_token.span,
                    format!("write `fn` instead of `{symbol}` to declare a function"),
                    "fn",
                    Applicability::MachineApplicable,
                );
            }
        }

        if let TokenKind::Ident(prev, _) = &self.prev_token.kind
            && let TokenKind::Ident(cur, _) = &self.token.kind
        {
            let concat = Symbol::intern(&format!("{prev}{cur}"));
            let ident = Ident::new(concat, DUMMY_SP);
            if ident.is_used_keyword() || ident.is_reserved() || ident.is_raw_guess() {
                let concat_span = self.prev_token.span.to(self.token.span);
                err.span_suggestion_verbose(
                    concat_span,
                    format!("consider removing the space to spell keyword `{concat}`"),
                    concat,
                    Applicability::MachineApplicable,
                );
            }
        }

        // Try to detect an intended c-string literal while using a pre-2021 edition. The heuristic
        // here is to identify a cooked, uninterpolated `c` id immediately followed by a string, or
        // a cooked, uninterpolated `cr` id immediately followed by a string or a `#`, in an edition
        // where c-string literals are not allowed. There is the very slight possibility of a false
        // positive for a `cr#` that wasn't intended to start a c-string literal, but identifying
        // that in the parser requires unbounded lookahead, so we only add a hint to the existing
        // error rather than replacing it entirely.
        if ((self.prev_token == TokenKind::Ident(sym::c, IdentIsRaw::No)
            && matches!(&self.token.kind, TokenKind::Literal(token::Lit { kind: token::Str, .. })))
            || (self.prev_token == TokenKind::Ident(sym::cr, IdentIsRaw::No)
                && matches!(
                    &self.token.kind,
                    TokenKind::Literal(token::Lit { kind: token::Str, .. }) | token::Pound
                )))
            && self.prev_token.span.hi() == self.token.span.lo()
            && !self.token.span.at_least_rust_2021()
        {
            err.note("you may be trying to write a c-string literal");
            err.note("c-string literals require Rust 2021 or later");
            err.subdiagnostic(HelpUseLatestEdition::new());
        }

        // `pub` may be used for an item or `pub(crate)`
        if self.prev_token.is_ident_named(sym::public)
            && (self.token.can_begin_item() || self.token == TokenKind::OpenParen)
        {
            err.span_suggestion_short(
                self.prev_token.span,
                "write `pub` instead of `public` to make the item public",
                "pub",
                Applicability::MachineApplicable,
            );
        }

        if let token::DocComment(kind, style, _) = self.token.kind {
            // This is to avoid suggesting converting a doc comment to a regular comment
            // when missing a comma before the doc comment in lists (#142311):
            //
            // ```
            // enum Foo{
            //     A /// xxxxxxx
            //     B,
            // }
            // ```
            if !expected.contains(&TokenType::Comma) {
                // We have something like `expr //!val` where the user likely meant `expr // !val`
                let pos = self.token.span.lo() + BytePos(2);
                let span = self.token.span.with_lo(pos).with_hi(pos);
                err.span_suggestion_verbose(
                    span,
                    format!(
                        "add a space before {} to write a regular comment",
                        match (kind, style) {
                            (token::CommentKind::Line, ast::AttrStyle::Inner) => "`!`",
                            (token::CommentKind::Block, ast::AttrStyle::Inner) => "`!`",
                            (token::CommentKind::Line, ast::AttrStyle::Outer) => "the last `/`",
                            (token::CommentKind::Block, ast::AttrStyle::Outer) => "the last `*`",
                        },
                    ),
                    " ".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        }

        let sp = if self.token == token::Eof {
            // This is EOF; don't want to point at the following char, but rather the last token.
            self.prev_token.span
        } else {
            label_sp
        };

        if self.check_too_many_raw_str_terminators(&mut err) {
            if expected.contains(&TokenType::Semi) && self.eat(exp!(Semi)) {
                let guar = err.emit();
                return Ok(guar);
            } else {
                return Err(err);
            }
        }

        if self.prev_token.span == DUMMY_SP {
            // Account for macro context where the previous span might not be
            // available to avoid incorrect output (#54841).
            err.span_label(self.token.span, label_exp);
        } else if !sm.is_multiline(self.token.span.shrink_to_hi().until(sp.shrink_to_lo())) {
            // When the spans are in the same line, it means that the only content between
            // them is whitespace, point at the found token in that case:
            //
            // X |     () => { syntax error };
            //   |                    ^^^^^ expected one of 8 possible tokens here
            //
            // instead of having:
            //
            // X |     () => { syntax error };
            //   |                   -^^^^^ unexpected token
            //   |                   |
            //   |                   expected one of 8 possible tokens here
            err.span_label(self.token.span, label_exp);
        } else {
            err.span_label(sp, label_exp);
            err.span_label(self.token.span, "unexpected token");
        }

        // Check for misspelled keywords if there are no suggestions added to the diagnostic.
        if matches!(&err.suggestions, Suggestions::Enabled(list) if list.is_empty()) {
            self.check_for_misspelled_kw(&mut err, &expected);
        }
        Err(err)
    }

    /// Adds a label when `&raw EXPR` was written instead of `&raw const EXPR`/`&raw mut EXPR`.
    ///
    /// Given that not all parser diagnostics flow through `expected_one_of_not_found`, this
    /// label may need added to other diagnostics emission paths as needed.
    pub(super) fn label_expected_raw_ref(&mut self, err: &mut Diag<'_>) {
        if self.prev_token.is_keyword(kw::Raw)
            && self.expected_token_types.contains(TokenType::KwMut)
            && self.expected_token_types.contains(TokenType::KwConst)
            && self.token.can_begin_expr()
        {
            err.span_suggestions(
                self.prev_token.span.shrink_to_hi(),
                "`&raw` must be followed by `const` or `mut` to be a raw reference expression",
                [" const".to_string(), " mut".to_string()],
                Applicability::MaybeIncorrect,
            );
        }
    }

    /// Checks if the current token or the previous token are misspelled keywords
    /// and adds a helpful suggestion.
    fn check_for_misspelled_kw(&self, err: &mut Diag<'_>, expected: &[TokenType]) {
        let Some((curr_ident, _)) = self.token.ident() else {
            return;
        };
        let expected_token_types: &[TokenType] =
            expected.len().checked_sub(10).map_or(&expected, |index| &expected[index..]);
        let expected_keywords: Vec<Symbol> =
            expected_token_types.iter().filter_map(|token| token.is_keyword()).collect();

        // When there are a few keywords in the last ten elements of `self.expected_token_types`
        // and the current token is an identifier, it's probably a misspelled keyword. This handles
        // code like `async Move {}`, misspelled `if` in match guard, misspelled `else` in
        // `if`-`else` and misspelled `where` in a where clause.
        if !expected_keywords.is_empty()
            && !curr_ident.is_used_keyword()
            && let Some(misspelled_kw) = find_similar_kw(curr_ident, &expected_keywords)
        {
            err.subdiagnostic(misspelled_kw);
            // We don't want other suggestions to be added as they are most likely meaningless
            // when there is a misspelled keyword.
            err.seal_suggestions();
        } else if let Some((prev_ident, _)) = self.prev_token.ident()
            && !prev_ident.is_used_keyword()
        {
            // We generate a list of all keywords at runtime rather than at compile time
            // so that it gets generated only when the diagnostic needs it.
            // Also, it is unlikely that this list is generated multiple times because the
            // parser halts after execution hits this path.
            let all_keywords = used_keywords(|| prev_ident.span.edition());

            // Otherwise, check the previous token with all the keywords as possible candidates.
            // This handles code like `Struct Human;` and `While a < b {}`.
            // We check the previous token only when the current token is an identifier to avoid
            // false positives like suggesting keyword `for` for `extern crate foo {}`.
            if let Some(misspelled_kw) = find_similar_kw(prev_ident, &all_keywords) {
                err.subdiagnostic(misspelled_kw);
                // We don't want other suggestions to be added as they are most likely meaningless
                // when there is a misspelled keyword.
                err.seal_suggestions();
            }
        }
    }

    /// The user has written `#[attr] expr` which is unsupported. (#106020)
    pub(super) fn attr_on_non_tail_expr(&self, expr: &Expr) -> ErrorGuaranteed {
        // Missing semicolon typo error.
        let span = self.prev_token.span.shrink_to_hi();
        let mut err = self.dcx().create_err(ExpectedSemi {
            span,
            token: self.token,
            unexpected_token_label: Some(self.token.span),
            sugg: ExpectedSemiSugg::AddSemi(span),
        });
        let attr_span = match &expr.attrs[..] {
            [] => unreachable!(),
            [only] => only.span,
            [first, rest @ ..] => {
                for attr in rest {
                    err.span_label(attr.span, "");
                }
                first.span
            }
        };
        err.span_label(
            attr_span,
            format!(
                "only `;` terminated statements or tail expressions are allowed after {}",
                if expr.attrs.len() == 1 { "this attribute" } else { "these attributes" },
            ),
        );
        if self.token == token::Pound && self.look_ahead(1, |t| *t == token::OpenBracket) {
            // We have
            // #[attr]
            // expr
            // #[not_attr]
            // other_expr
            err.span_label(span, "expected `;` here");
            err.multipart_suggestion(
                "alternatively, consider surrounding the expression with a block",
                vec![
                    (expr.span.shrink_to_lo(), "{ ".to_string()),
                    (expr.span.shrink_to_hi(), " }".to_string()),
                ],
                Applicability::MachineApplicable,
            );

            // Special handling for `#[cfg(...)]` chains
            let mut snapshot = self.create_snapshot_for_diagnostic();
            if let [attr] = &expr.attrs[..]
                && let ast::AttrKind::Normal(attr_kind) = &attr.kind
                && let [segment] = &attr_kind.item.path.segments[..]
                && segment.ident.name == sym::cfg
                && let Some(args_span) = attr_kind.item.args.span()
                && let next_attr = match snapshot.parse_attribute(InnerAttrPolicy::Forbidden(None))
                {
                    Ok(next_attr) => next_attr,
                    Err(inner_err) => {
                        inner_err.cancel();
                        return err.emit();
                    }
                }
                && let ast::AttrKind::Normal(next_attr_kind) = next_attr.kind
                && let Some(next_attr_args_span) = next_attr_kind.item.args.span()
                && let [next_segment] = &next_attr_kind.item.path.segments[..]
                && segment.ident.name == sym::cfg
            {
                let next_expr = match snapshot.parse_expr() {
                    Ok(next_expr) => next_expr,
                    Err(inner_err) => {
                        inner_err.cancel();
                        return err.emit();
                    }
                };
                // We have for sure
                // #[cfg(..)]
                // expr
                // #[cfg(..)]
                // other_expr
                // So we suggest using `if cfg!(..) { expr } else if cfg!(..) { other_expr }`.
                let margin = self.psess.source_map().span_to_margin(next_expr.span).unwrap_or(0);
                let sugg = vec![
                    (attr.span.with_hi(segment.span().hi()), "if cfg!".to_string()),
                    (args_span.shrink_to_hi().with_hi(attr.span.hi()), " {".to_string()),
                    (expr.span.shrink_to_lo(), "    ".to_string()),
                    (
                        next_attr.span.with_hi(next_segment.span().hi()),
                        "} else if cfg!".to_string(),
                    ),
                    (
                        next_attr_args_span.shrink_to_hi().with_hi(next_attr.span.hi()),
                        " {".to_string(),
                    ),
                    (next_expr.span.shrink_to_lo(), "    ".to_string()),
                    (next_expr.span.shrink_to_hi(), format!("\n{}}}", " ".repeat(margin))),
                ];
                err.multipart_suggestion(
                    "it seems like you are trying to provide different expressions depending on \
                     `cfg`, consider using `if cfg!(..)`",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        }

        err.emit()
    }

    fn check_too_many_raw_str_terminators(&mut self, err: &mut Diag<'_>) -> bool {
        let sm = self.psess.source_map();
        match (&self.prev_token.kind, &self.token.kind) {
            (
                TokenKind::Literal(Lit {
                    kind: LitKind::StrRaw(n_hashes) | LitKind::ByteStrRaw(n_hashes),
                    ..
                }),
                TokenKind::Pound,
            ) if !sm.is_multiline(
                self.prev_token.span.shrink_to_hi().until(self.token.span.shrink_to_lo()),
            ) =>
            {
                let n_hashes: u8 = *n_hashes;
                err.primary_message("too many `#` when terminating raw string");
                let str_span = self.prev_token.span;
                let mut span = self.token.span;
                let mut count = 0;
                while self.token == TokenKind::Pound
                    && !sm.is_multiline(span.shrink_to_hi().until(self.token.span.shrink_to_lo()))
                {
                    span = span.with_hi(self.token.span.hi());
                    self.bump();
                    count += 1;
                }
                err.span(span);
                err.span_suggestion(
                    span,
                    format!("remove the extra `#`{}", pluralize!(count)),
                    "",
                    Applicability::MachineApplicable,
                );
                err.span_label(
                    str_span,
                    format!("this raw string started with {n_hashes} `#`{}", pluralize!(n_hashes)),
                );
                true
            }
            _ => false,
        }
    }

    pub(super) fn maybe_suggest_struct_literal(
        &mut self,
        lo: Span,
        s: BlockCheckMode,
        maybe_struct_name: token::Token,
    ) -> Option<PResult<'a, P<Block>>> {
        if self.token.is_ident() && self.look_ahead(1, |t| t == &token::Colon) {
            // We might be having a struct literal where people forgot to include the path:
            // fn foo() -> Foo {
            //     field: value,
            // }
            debug!(?maybe_struct_name, ?self.token);
            let mut snapshot = self.create_snapshot_for_diagnostic();
            let path = Path {
                segments: ThinVec::new(),
                span: self.prev_token.span.shrink_to_lo(),
                tokens: None,
            };
            let struct_expr = snapshot.parse_expr_struct(None, path, false);
            let block_tail = self.parse_block_tail(lo, s, AttemptLocalParseRecovery::No);
            return Some(match (struct_expr, block_tail) {
                (Ok(expr), Err(err)) => {
                    // We have encountered the following:
                    // fn foo() -> Foo {
                    //     field: value,
                    // }
                    // Suggest:
                    // fn foo() -> Foo { Path {
                    //     field: value,
                    // } }
                    err.cancel();
                    self.restore_snapshot(snapshot);
                    let guar = self.dcx().emit_err(StructLiteralBodyWithoutPath {
                        span: expr.span,
                        sugg: StructLiteralBodyWithoutPathSugg {
                            before: expr.span.shrink_to_lo(),
                            after: expr.span.shrink_to_hi(),
                        },
                    });
                    Ok(self.mk_block(
                        thin_vec![self.mk_stmt_err(expr.span, guar)],
                        s,
                        lo.to(self.prev_token.span),
                    ))
                }
                (Err(err), Ok(tail)) => {
                    // We have a block tail that contains a somehow valid expr.
                    err.cancel();
                    Ok(tail)
                }
                (Err(snapshot_err), Err(err)) => {
                    // We don't know what went wrong, emit the normal error.
                    snapshot_err.cancel();
                    self.consume_block(exp!(OpenBrace), exp!(CloseBrace), ConsumeClosingDelim::Yes);
                    Err(err)
                }
                (Ok(_), Ok(tail)) => Ok(tail),
            });
        }
        None
    }

    pub(super) fn recover_closure_body(
        &mut self,
        mut err: Diag<'a>,
        before: token::Token,
        prev: token::Token,
        token: token::Token,
        lo: Span,
        decl_hi: Span,
    ) -> PResult<'a, P<Expr>> {
        err.span_label(lo.to(decl_hi), "while parsing the body of this closure");
        let guar = match before.kind {
            token::OpenBrace if token.kind != token::OpenBrace => {
                // `{ || () }` should have been `|| { () }`
                err.multipart_suggestion(
                    "you might have meant to open the body of the closure, instead of enclosing \
                     the closure in a block",
                    vec![
                        (before.span, String::new()),
                        (prev.span.shrink_to_hi(), " {".to_string()),
                    ],
                    Applicability::MaybeIncorrect,
                );
                let guar = err.emit();
                self.eat_to_tokens(&[exp!(CloseBrace)]);
                guar
            }
            token::OpenParen if token.kind != token::OpenBrace => {
                // We are within a function call or tuple, we can emit the error
                // and recover.
                self.eat_to_tokens(&[exp!(CloseParen), exp!(Comma)]);

                err.multipart_suggestion_verbose(
                    "you might have meant to open the body of the closure",
                    vec![
                        (prev.span.shrink_to_hi(), " {".to_string()),
                        (self.token.span.shrink_to_lo(), "}".to_string()),
                    ],
                    Applicability::MaybeIncorrect,
                );
                err.emit()
            }
            _ if token.kind != token::OpenBrace => {
                // We don't have a heuristic to correctly identify where the block
                // should be closed.
                err.multipart_suggestion_verbose(
                    "you might have meant to open the body of the closure",
                    vec![(prev.span.shrink_to_hi(), " {".to_string())],
                    Applicability::HasPlaceholders,
                );
                return Err(err);
            }
            _ => return Err(err),
        };
        Ok(self.mk_expr_err(lo.to(self.token.span), guar))
    }

    /// Eats and discards tokens until one of `closes` is encountered. Respects token trees,
    /// passes through any errors encountered. Used for error recovery.
    pub(super) fn eat_to_tokens(&mut self, closes: &[ExpTokenPair<'_>]) {
        if let Err(err) = self
            .parse_seq_to_before_tokens(closes, &[], SeqSep::none(), |p| Ok(p.parse_token_tree()))
        {
            err.cancel();
        }
    }

    /// This function checks if there are trailing angle brackets and produces
    /// a diagnostic to suggest removing them.
    ///
    /// ```ignore (diagnostic)
    /// let _ = [1, 2, 3].into_iter().collect::<Vec<usize>>>>();
    ///                                                    ^^ help: remove extra angle brackets
    /// ```
    ///
    /// If `true` is returned, then trailing brackets were recovered, tokens were consumed
    /// up until one of the tokens in 'end' was encountered, and an error was emitted.
    pub(super) fn check_trailing_angle_brackets(
        &mut self,
        segment: &PathSegment,
        end: &[ExpTokenPair<'_>],
    ) -> Option<ErrorGuaranteed> {
        if !self.may_recover() {
            return None;
        }

        // This function is intended to be invoked after parsing a path segment where there are two
        // cases:
        //
        // 1. A specific token is expected after the path segment.
        //    eg. `x.foo(`, `x.foo::<u32>(` (parenthesis - method call),
        //        `Foo::`, or `Foo::<Bar>::` (mod sep - continued path).
        // 2. No specific token is expected after the path segment.
        //    eg. `x.foo` (field access)
        //
        // This function is called after parsing `.foo` and before parsing the token `end` (if
        // present). This includes any angle bracket arguments, such as `.foo::<u32>` or
        // `Foo::<Bar>`.

        // We only care about trailing angle brackets if we previously parsed angle bracket
        // arguments. This helps stop us incorrectly suggesting that extra angle brackets be
        // removed in this case:
        //
        // `x.foo >> (3)` (where `x.foo` is a `u32` for example)
        //
        // This case is particularly tricky as we won't notice it just looking at the tokens -
        // it will appear the same (in terms of upcoming tokens) as below (since the `::<u32>` will
        // have already been parsed):
        //
        // `x.foo::<u32>>>(3)`
        let parsed_angle_bracket_args =
            segment.args.as_ref().is_some_and(|args| args.is_angle_bracketed());

        debug!(
            "check_trailing_angle_brackets: parsed_angle_bracket_args={:?}",
            parsed_angle_bracket_args,
        );
        if !parsed_angle_bracket_args {
            return None;
        }

        // Keep the span at the start so we can highlight the sequence of `>` characters to be
        // removed.
        let lo = self.token.span;

        // We need to look-ahead to see if we have `>` characters without moving the cursor forward
        // (since we might have the field access case and the characters we're eating are
        // actual operators and not trailing characters - ie `x.foo >> 3`).
        let mut position = 0;

        // We can encounter `>` or `>>` tokens in any order, so we need to keep track of how
        // many of each (so we can correctly pluralize our error messages) and continue to
        // advance.
        let mut number_of_shr = 0;
        let mut number_of_gt = 0;
        while self.look_ahead(position, |t| {
            trace!("check_trailing_angle_brackets: t={:?}", t);
            if *t == token::Shr {
                number_of_shr += 1;
                true
            } else if *t == token::Gt {
                number_of_gt += 1;
                true
            } else {
                false
            }
        }) {
            position += 1;
        }

        // If we didn't find any trailing `>` characters, then we have nothing to error about.
        debug!(
            "check_trailing_angle_brackets: number_of_gt={:?} number_of_shr={:?}",
            number_of_gt, number_of_shr,
        );
        if number_of_gt < 1 && number_of_shr < 1 {
            return None;
        }

        // Finally, double check that we have our end token as otherwise this is the
        // second case.
        if self.look_ahead(position, |t| {
            trace!("check_trailing_angle_brackets: t={:?}", t);
            end.iter().any(|exp| exp.tok == &t.kind)
        }) {
            // Eat from where we started until the end token so that parsing can continue
            // as if we didn't have those extra angle brackets.
            self.eat_to_tokens(end);
            let span = lo.to(self.prev_token.span);

            let num_extra_brackets = number_of_gt + number_of_shr * 2;
            return Some(self.dcx().emit_err(UnmatchedAngleBrackets { span, num_extra_brackets }));
        }
        None
    }

    /// Check if a method call with an intended turbofish has been written without surrounding
    /// angle brackets.
    pub(super) fn check_turbofish_missing_angle_brackets(&mut self, segment: &mut PathSegment) {
        if !self.may_recover() {
            return;
        }

        if self.token == token::PathSep && segment.args.is_none() {
            let snapshot = self.create_snapshot_for_diagnostic();
            self.bump();
            let lo = self.token.span;
            match self.parse_angle_args(None) {
                Ok(args) => {
                    let span = lo.to(self.prev_token.span);
                    // Detect trailing `>` like in `x.collect::Vec<_>>()`.
                    let mut trailing_span = self.prev_token.span.shrink_to_hi();
                    while self.token == token::Shr || self.token == token::Gt {
                        trailing_span = trailing_span.to(self.token.span);
                        self.bump();
                    }
                    if self.token == token::OpenParen {
                        // Recover from bad turbofish: `foo.collect::Vec<_>()`.
                        segment.args = Some(AngleBracketedArgs { args, span }.into());

                        self.dcx().emit_err(GenericParamsWithoutAngleBrackets {
                            span,
                            sugg: GenericParamsWithoutAngleBracketsSugg {
                                left: span.shrink_to_lo(),
                                right: trailing_span,
                            },
                        });
                    } else {
                        // This doesn't look like an invalid turbofish, can't recover parse state.
                        self.restore_snapshot(snapshot);
                    }
                }
                Err(err) => {
                    // We couldn't parse generic parameters, unlikely to be a turbofish. Rely on
                    // generic parse error instead.
                    err.cancel();
                    self.restore_snapshot(snapshot);
                }
            }
        }
    }

    /// When writing a turbofish with multiple type parameters missing the leading `::`, we will
    /// encounter a parse error when encountering the first `,`.
    pub(super) fn check_mistyped_turbofish_with_multiple_type_params(
        &mut self,
        mut e: Diag<'a>,
        expr: &mut P<Expr>,
    ) -> PResult<'a, ErrorGuaranteed> {
        if let ExprKind::Binary(binop, _, _) = &expr.kind
            && let ast::BinOpKind::Lt = binop.node
            && self.eat(exp!(Comma))
        {
            let x = self.parse_seq_to_before_end(
                exp!(Gt),
                SeqSep::trailing_allowed(exp!(Comma)),
                |p| match p.parse_generic_arg(None)? {
                    Some(arg) => Ok(arg),
                    // If we didn't eat a generic arg, then we should error.
                    None => p.unexpected_any(),
                },
            );
            match x {
                Ok((_, _, Recovered::No)) => {
                    if self.eat(exp!(Gt)) {
                        // We made sense of it. Improve the error message.
                        e.span_suggestion_verbose(
                            binop.span.shrink_to_lo(),
                            fluent::parse_sugg_turbofish_syntax,
                            "::",
                            Applicability::MaybeIncorrect,
                        );
                        match self.parse_expr() {
                            Ok(_) => {
                                // The subsequent expression is valid. Mark
                                // `expr` as erroneous and emit `e` now, but
                                // return `Ok` so parsing can continue.
                                let guar = e.emit();
                                *expr = self.mk_expr_err(expr.span.to(self.prev_token.span), guar);
                                return Ok(guar);
                            }
                            Err(err) => {
                                err.cancel();
                            }
                        }
                    }
                }
                Ok((_, _, Recovered::Yes(_))) => {}
                Err(err) => {
                    err.cancel();
                }
            }
        }
        Err(e)
    }

    /// Suggest add the missing `let` before the identifier in stmt
    /// `a: Ty = 1` -> `let a: Ty = 1`
    pub(super) fn suggest_add_missing_let_for_stmt(&mut self, err: &mut Diag<'a>) {
        if self.token == token::Colon {
            let prev_span = self.prev_token.span.shrink_to_lo();
            let snapshot = self.create_snapshot_for_diagnostic();
            self.bump();
            match self.parse_ty() {
                Ok(_) => {
                    if self.token == token::Eq {
                        let sugg = SuggAddMissingLetStmt { span: prev_span };
                        sugg.add_to_diag(err);
                    }
                }
                Err(e) => {
                    e.cancel();
                }
            }
            self.restore_snapshot(snapshot);
        }
    }

    /// Check to see if a pair of chained operators looks like an attempt at chained comparison,
    /// e.g. `1 < x <= 3`. If so, suggest either splitting the comparison into two, or
    /// parenthesising the leftmost comparison. The return value indicates if recovery happened.
    fn attempt_chained_comparison_suggestion(
        &mut self,
        err: &mut ComparisonOperatorsCannotBeChained,
        inner_op: &Expr,
        outer_op: &Spanned<AssocOp>,
    ) -> bool {
        if let ExprKind::Binary(op, l1, r1) = &inner_op.kind {
            if let ExprKind::Field(_, ident) = l1.kind
                && !ident.is_numeric()
                && !matches!(r1.kind, ExprKind::Lit(_))
            {
                // The parser has encountered `foo.bar<baz`, the likelihood of the turbofish
                // suggestion being the only one to apply is high.
                return false;
            }
            return match (op.node, &outer_op.node) {
                // `x == y == z`
                (BinOpKind::Eq, AssocOp::Binary(BinOpKind::Eq)) |
                // `x < y < z` and friends.
                (BinOpKind::Lt, AssocOp::Binary(BinOpKind::Lt | BinOpKind::Le)) |
                (BinOpKind::Le, AssocOp::Binary(BinOpKind::Lt | BinOpKind::Le)) |
                // `x > y > z` and friends.
                (BinOpKind::Gt, AssocOp::Binary(BinOpKind::Gt | BinOpKind::Ge)) |
                (BinOpKind::Ge, AssocOp::Binary(BinOpKind::Gt | BinOpKind::Ge)) => {
                    let expr_to_str = |e: &Expr| {
                        self.span_to_snippet(e.span)
                            .unwrap_or_else(|_| pprust::expr_to_string(e))
                    };
                    err.chaining_sugg = Some(ComparisonOperatorsCannotBeChainedSugg::SplitComparison {
                        span: inner_op.span.shrink_to_hi(),
                        middle_term: expr_to_str(r1),
                    });
                    false // Keep the current parse behavior, where the AST is `(x < y) < z`.
                }
                // `x == y < z`
                (
                    BinOpKind::Eq,
                    AssocOp::Binary(BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge)
                ) => {
                    // Consume `z`/outer-op-rhs.
                    let snapshot = self.create_snapshot_for_diagnostic();
                    match self.parse_expr() {
                        Ok(r2) => {
                            // We are sure that outer-op-rhs could be consumed, the suggestion is
                            // likely correct.
                            err.chaining_sugg = Some(ComparisonOperatorsCannotBeChainedSugg::Parenthesize {
                                left: r1.span.shrink_to_lo(),
                                right: r2.span.shrink_to_hi(),
                            });
                            true
                        }
                        Err(expr_err) => {
                            expr_err.cancel();
                            self.restore_snapshot(snapshot);
                            true
                        }
                    }
                }
                // `x > y == z`
                (
                    BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge,
                    AssocOp::Binary(BinOpKind::Eq)
                ) => {
                    let snapshot = self.create_snapshot_for_diagnostic();
                    // At this point it is always valid to enclose the lhs in parentheses, no
                    // further checks are necessary.
                    match self.parse_expr() {
                        Ok(_) => {
                            err.chaining_sugg = Some(ComparisonOperatorsCannotBeChainedSugg::Parenthesize {
                                left: l1.span.shrink_to_lo(),
                                right: r1.span.shrink_to_hi(),
                            });
                            true
                        }
                        Err(expr_err) => {
                            expr_err.cancel();
                            self.restore_snapshot(snapshot);
                            false
                        }
                    }
                }
                _ => false
            };
        }
        false
    }

    /// Produces an error if comparison operators are chained (RFC #558).
    /// We only need to check the LHS, not the RHS, because all comparison ops have same
    /// precedence (see `fn precedence`) and are left-associative (see `fn fixity`).
    ///
    /// This can also be hit if someone incorrectly writes `foo<bar>()` when they should have used
    /// the turbofish (`foo::<bar>()`) syntax. We attempt some heuristic recovery if that is the
    /// case.
    ///
    /// Keep in mind that given that `outer_op.is_comparison()` holds and comparison ops are left
    /// associative we can infer that we have:
    ///
    /// ```text
    ///           outer_op
    ///           /   \
    ///     inner_op   r2
    ///        /  \
    ///      l1    r1
    /// ```
    pub(super) fn check_no_chained_comparison(
        &mut self,
        inner_op: &Expr,
        outer_op: &Spanned<AssocOp>,
    ) -> PResult<'a, Option<P<Expr>>> {
        debug_assert!(
            outer_op.node.is_comparison(),
            "check_no_chained_comparison: {:?} is not comparison",
            outer_op.node,
        );

        let mk_err_expr =
            |this: &Self, span, guar| Ok(Some(this.mk_expr(span, ExprKind::Err(guar))));

        match &inner_op.kind {
            ExprKind::Binary(op, l1, r1) if op.node.is_comparison() => {
                let mut err = ComparisonOperatorsCannotBeChained {
                    span: vec![op.span, self.prev_token.span],
                    suggest_turbofish: None,
                    help_turbofish: false,
                    chaining_sugg: None,
                };

                // Include `<` to provide this recommendation even in a case like
                // `Foo<Bar<Baz<Qux, ()>>>`
                if op.node == BinOpKind::Lt && outer_op.node == AssocOp::Binary(BinOpKind::Lt)
                    || outer_op.node == AssocOp::Binary(BinOpKind::Gt)
                {
                    if outer_op.node == AssocOp::Binary(BinOpKind::Lt) {
                        let snapshot = self.create_snapshot_for_diagnostic();
                        self.bump();
                        // So far we have parsed `foo<bar<`, consume the rest of the type args.
                        let modifiers = [(token::Lt, 1), (token::Gt, -1), (token::Shr, -2)];
                        self.consume_tts(1, &modifiers);

                        if !matches!(self.token.kind, token::OpenParen | token::PathSep) {
                            // We don't have `foo< bar >(` or `foo< bar >::`, so we rewind the
                            // parser and bail out.
                            self.restore_snapshot(snapshot);
                        }
                    }
                    return if self.token == token::PathSep {
                        // We have some certainty that this was a bad turbofish at this point.
                        // `foo< bar >::`
                        if let ExprKind::Binary(o, ..) = inner_op.kind
                            && o.node == BinOpKind::Lt
                        {
                            err.suggest_turbofish = Some(op.span.shrink_to_lo());
                        } else {
                            err.help_turbofish = true;
                        }

                        let snapshot = self.create_snapshot_for_diagnostic();
                        self.bump(); // `::`

                        // Consume the rest of the likely `foo<bar>::new()` or return at `foo<bar>`.
                        match self.parse_expr() {
                            Ok(_) => {
                                // 99% certain that the suggestion is correct, continue parsing.
                                let guar = self.dcx().emit_err(err);
                                // FIXME: actually check that the two expressions in the binop are
                                // paths and resynthesize new fn call expression instead of using
                                // `ExprKind::Err` placeholder.
                                mk_err_expr(self, inner_op.span.to(self.prev_token.span), guar)
                            }
                            Err(expr_err) => {
                                expr_err.cancel();
                                // Not entirely sure now, but we bubble the error up with the
                                // suggestion.
                                self.restore_snapshot(snapshot);
                                Err(self.dcx().create_err(err))
                            }
                        }
                    } else if self.token == token::OpenParen {
                        // We have high certainty that this was a bad turbofish at this point.
                        // `foo< bar >(`
                        if let ExprKind::Binary(o, ..) = inner_op.kind
                            && o.node == BinOpKind::Lt
                        {
                            err.suggest_turbofish = Some(op.span.shrink_to_lo());
                        } else {
                            err.help_turbofish = true;
                        }
                        // Consume the fn call arguments.
                        match self.consume_fn_args() {
                            Err(()) => Err(self.dcx().create_err(err)),
                            Ok(()) => {
                                let guar = self.dcx().emit_err(err);
                                // FIXME: actually check that the two expressions in the binop are
                                // paths and resynthesize new fn call expression instead of using
                                // `ExprKind::Err` placeholder.
                                mk_err_expr(self, inner_op.span.to(self.prev_token.span), guar)
                            }
                        }
                    } else {
                        if !matches!(l1.kind, ExprKind::Lit(_))
                            && !matches!(r1.kind, ExprKind::Lit(_))
                        {
                            // All we know is that this is `foo < bar >` and *nothing* else. Try to
                            // be helpful, but don't attempt to recover.
                            err.help_turbofish = true;
                        }

                        // If it looks like a genuine attempt to chain operators (as opposed to a
                        // misformatted turbofish, for instance), suggest a correct form.
                        let recovered = self
                            .attempt_chained_comparison_suggestion(&mut err, inner_op, outer_op);
                        if recovered {
                            let guar = self.dcx().emit_err(err);
                            mk_err_expr(self, inner_op.span.to(self.prev_token.span), guar)
                        } else {
                            // These cases cause too many knock-down errors, bail out (#61329).
                            Err(self.dcx().create_err(err))
                        }
                    };
                }
                let recovered =
                    self.attempt_chained_comparison_suggestion(&mut err, inner_op, outer_op);
                let guar = self.dcx().emit_err(err);
                if recovered {
                    return mk_err_expr(self, inner_op.span.to(self.prev_token.span), guar);
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn consume_fn_args(&mut self) -> Result<(), ()> {
        let snapshot = self.create_snapshot_for_diagnostic();
        self.bump(); // `(`

        // Consume the fn call arguments.
        let modifiers = [(token::OpenParen, 1), (token::CloseParen, -1)];
        self.consume_tts(1, &modifiers);

        if self.token == token::Eof {
            // Not entirely sure that what we consumed were fn arguments, rollback.
            self.restore_snapshot(snapshot);
            Err(())
        } else {
            // 99% certain that the suggestion is correct, continue parsing.
            Ok(())
        }
    }

    pub(super) fn maybe_report_ambiguous_plus(&mut self, impl_dyn_multi: bool, ty: &Ty) {
        if impl_dyn_multi {
            self.dcx().emit_err(AmbiguousPlus {
                span: ty.span,
                suggestion: AddParen { lo: ty.span.shrink_to_lo(), hi: ty.span.shrink_to_hi() },
            });
        }
    }

    /// Swift lets users write `Ty?` to mean `Option<Ty>`. Parse the construct and recover from it.
    pub(super) fn maybe_recover_from_question_mark(&mut self, ty: P<Ty>) -> P<Ty> {
        if self.token == token::Question {
            self.bump();
            let guar = self.dcx().emit_err(QuestionMarkInType {
                span: self.prev_token.span,
                sugg: QuestionMarkInTypeSugg {
                    left: ty.span.shrink_to_lo(),
                    right: self.prev_token.span,
                },
            });
            self.mk_ty(ty.span.to(self.prev_token.span), TyKind::Err(guar))
        } else {
            ty
        }
    }

    /// Rust has no ternary operator (`cond ? then : else`). Parse it and try
    /// to recover from it if `then` and `else` are valid expressions. Returns
    /// an err if this appears to be a ternary expression.
    /// If we have the span of the condition, we can provide a better error span
    /// and code suggestion.
    pub(super) fn maybe_recover_from_ternary_operator(
        &mut self,
        cond: Option<Span>,
    ) -> PResult<'a, ()> {
        if self.prev_token != token::Question {
            return PResult::Ok(());
        }

        let question = self.prev_token.span;
        let lo = cond.unwrap_or(question).lo();
        let snapshot = self.create_snapshot_for_diagnostic();

        if match self.parse_expr() {
            Ok(_) => true,
            Err(err) => {
                err.cancel();
                // The colon can sometimes be mistaken for type
                // ascription. Catch when this happens and continue.
                self.token == token::Colon
            }
        } {
            if self.eat_noexpect(&token::Colon) {
                let colon = self.prev_token.span;
                match self.parse_expr() {
                    Ok(expr) => {
                        let sugg = cond.map(|cond| TernaryOperatorSuggestion {
                            before_cond: cond.shrink_to_lo(),
                            question,
                            colon,
                            end: expr.span.shrink_to_hi(),
                        });
                        return Err(self.dcx().create_err(TernaryOperator {
                            span: self.prev_token.span.with_lo(lo),
                            sugg,
                            no_sugg: sugg.is_none(),
                        }));
                    }
                    Err(err) => {
                        err.cancel();
                    }
                };
            }
        }
        self.restore_snapshot(snapshot);
        Ok(())
    }

    pub(super) fn maybe_recover_from_bad_type_plus(&mut self, ty: &Ty) -> PResult<'a, ()> {
        // Do not add `+` to expected tokens.
        if !self.token.is_like_plus() {
            return Ok(());
        }

        self.bump(); // `+`
        let _bounds = self.parse_generic_bounds()?;
        let sub = match &ty.kind {
            TyKind::Ref(_lifetime, mut_ty) => {
                let lo = mut_ty.ty.span.shrink_to_lo();
                let hi = self.prev_token.span.shrink_to_hi();
                BadTypePlusSub::AddParen { suggestion: AddParen { lo, hi } }
            }
            TyKind::Ptr(..) | TyKind::BareFn(..) => {
                BadTypePlusSub::ForgotParen { span: ty.span.to(self.prev_token.span) }
            }
            _ => BadTypePlusSub::ExpectPath { span: ty.span },
        };

        self.dcx().emit_err(BadTypePlus { span: ty.span, sub });

        Ok(())
    }

    pub(super) fn recover_from_prefix_increment(
        &mut self,
        operand_expr: P<Expr>,
        op_span: Span,
        start_stmt: bool,
    ) -> PResult<'a, P<Expr>> {
        let standalone = if start_stmt { IsStandalone::Standalone } else { IsStandalone::Subexpr };
        let kind = IncDecRecovery { standalone, op: IncOrDec::Inc, fixity: UnaryFixity::Pre };
        self.recover_from_inc_dec(operand_expr, kind, op_span)
    }

    pub(super) fn recover_from_postfix_increment(
        &mut self,
        operand_expr: P<Expr>,
        op_span: Span,
        start_stmt: bool,
    ) -> PResult<'a, P<Expr>> {
        let kind = IncDecRecovery {
            standalone: if start_stmt { IsStandalone::Standalone } else { IsStandalone::Subexpr },
            op: IncOrDec::Inc,
            fixity: UnaryFixity::Post,
        };
        self.recover_from_inc_dec(operand_expr, kind, op_span)
    }

    pub(super) fn recover_from_postfix_decrement(
        &mut self,
        operand_expr: P<Expr>,
        op_span: Span,
        start_stmt: bool,
    ) -> PResult<'a, P<Expr>> {
        let kind = IncDecRecovery {
            standalone: if start_stmt { IsStandalone::Standalone } else { IsStandalone::Subexpr },
            op: IncOrDec::Dec,
            fixity: UnaryFixity::Post,
        };
        self.recover_from_inc_dec(operand_expr, kind, op_span)
    }

    fn recover_from_inc_dec(
        &mut self,
        base: P<Expr>,
        kind: IncDecRecovery,
        op_span: Span,
    ) -> PResult<'a, P<Expr>> {
        let mut err = self.dcx().struct_span_err(
            op_span,
            format!("Rust has no {} {} operator", kind.fixity, kind.op.name()),
        );
        err.span_label(op_span, format!("not a valid {} operator", kind.fixity));

        let help_base_case = |mut err: Diag<'_, _>, base| {
            err.help(format!("use `{}= 1` instead", kind.op.chr()));
            err.emit();
            Ok(base)
        };

        // (pre, post)
        let spans = match kind.fixity {
            UnaryFixity::Pre => (op_span, base.span.shrink_to_hi()),
            UnaryFixity::Post => (base.span.shrink_to_lo(), op_span),
        };

        match kind.standalone {
            IsStandalone::Standalone => {
                self.inc_dec_standalone_suggest(kind, spans).emit_verbose(&mut err)
            }
            IsStandalone::Subexpr => {
                let Ok(base_src) = self.span_to_snippet(base.span) else {
                    return help_base_case(err, base);
                };
                match kind.fixity {
                    UnaryFixity::Pre => {
                        self.prefix_inc_dec_suggest(base_src, kind, spans).emit(&mut err)
                    }
                    UnaryFixity::Post => {
                        // won't suggest since we can not handle the precedences
                        // for example: `a + b++` has been parsed (a + b)++ and we can not suggest here
                        if !matches!(base.kind, ExprKind::Binary(_, _, _)) {
                            self.postfix_inc_dec_suggest(base_src, kind, spans).emit(&mut err)
                        }
                    }
                }
            }
        }
        Err(err)
    }

    fn prefix_inc_dec_suggest(
        &mut self,
        base_src: String,
        kind: IncDecRecovery,
        (pre_span, post_span): (Span, Span),
    ) -> MultiSugg {
        MultiSugg {
            msg: format!("use `{}= 1` instead", kind.op.chr()),
            patches: vec![
                (pre_span, "{ ".to_string()),
                (post_span, format!(" {}= 1; {} }}", kind.op.chr(), base_src)),
            ],
            applicability: Applicability::MachineApplicable,
        }
    }

    fn postfix_inc_dec_suggest(
        &mut self,
        base_src: String,
        kind: IncDecRecovery,
        (pre_span, post_span): (Span, Span),
    ) -> MultiSugg {
        let tmp_var = if base_src.trim() == "tmp" { "tmp_" } else { "tmp" };
        MultiSugg {
            msg: format!("use `{}= 1` instead", kind.op.chr()),
            patches: vec![
                (pre_span, format!("{{ let {tmp_var} = ")),
                (post_span, format!("; {} {}= 1; {} }}", base_src, kind.op.chr(), tmp_var)),
            ],
            applicability: Applicability::HasPlaceholders,
        }
    }

    fn inc_dec_standalone_suggest(
        &mut self,
        kind: IncDecRecovery,
        (pre_span, post_span): (Span, Span),
    ) -> MultiSugg {
        let mut patches = Vec::new();

        if !pre_span.is_empty() {
            patches.push((pre_span, String::new()));
        }

        patches.push((post_span, format!(" {}= 1", kind.op.chr())));
        MultiSugg {
            msg: format!("use `{}= 1` instead", kind.op.chr()),
            patches,
            applicability: Applicability::MachineApplicable,
        }
    }

    /// Tries to recover from associated item paths like `[T]::AssocItem` / `(T, U)::AssocItem`.
    /// Attempts to convert the base expression/pattern/type into a type, parses the `::AssocItem`
    /// tail, and combines them into a `<Ty>::AssocItem` expression/pattern/type.
    pub(super) fn maybe_recover_from_bad_qpath<T: RecoverQPath>(
        &mut self,
        base: P<T>,
    ) -> PResult<'a, P<T>> {
        if !self.may_recover() {
            return Ok(base);
        }

        // Do not add `::` to expected tokens.
        if self.token == token::PathSep {
            if let Some(ty) = base.to_ty() {
                return self.maybe_recover_from_bad_qpath_stage_2(ty.span, ty);
            }
        }
        Ok(base)
    }

    /// Given an already parsed `Ty`, parses the `::AssocItem` tail and
    /// combines them into a `<Ty>::AssocItem` expression/pattern/type.
    pub(super) fn maybe_recover_from_bad_qpath_stage_2<T: RecoverQPath>(
        &mut self,
        ty_span: Span,
        ty: P<Ty>,
    ) -> PResult<'a, P<T>> {
        self.expect(exp!(PathSep))?;

        let mut path = ast::Path { segments: ThinVec::new(), span: DUMMY_SP, tokens: None };
        self.parse_path_segments(&mut path.segments, T::PATH_STYLE, None)?;
        path.span = ty_span.to(self.prev_token.span);

        self.dcx().emit_err(BadQPathStage2 {
            span: ty_span,
            wrap: WrapType { lo: ty_span.shrink_to_lo(), hi: ty_span.shrink_to_hi() },
        });

        let path_span = ty_span.shrink_to_hi(); // Use an empty path since `position == 0`.
        Ok(P(T::recovered(Some(P(QSelf { ty, path_span, position: 0 })), path)))
    }

    /// This function gets called in places where a semicolon is NOT expected and if there's a
    /// semicolon it emits the appropriate error and returns true.
    pub fn maybe_consume_incorrect_semicolon(&mut self, previous_item: Option<&Item>) -> bool {
        if self.token != TokenKind::Semi {
            return false;
        }

        // Check previous item to add it to the diagnostic, for example to say
        // `enum declarations are not followed by a semicolon`
        let err = match previous_item {
            Some(previous_item) => {
                let name = match previous_item.kind {
                    // Say "braced struct" because tuple-structs and
                    // braceless-empty-struct declarations do take a semicolon.
                    ItemKind::Struct(..) => "braced struct",
                    _ => previous_item.kind.descr(),
                };
                IncorrectSemicolon { span: self.token.span, name, show_help: true }
            }
            None => IncorrectSemicolon { span: self.token.span, name: "", show_help: false },
        };
        self.dcx().emit_err(err);

        self.bump();
        true
    }

    /// Creates a `Diag` for an unexpected token `t` and tries to recover if it is a
    /// closing delimiter.
    pub(super) fn unexpected_try_recover(&mut self, t: &TokenKind) -> PResult<'a, Recovered> {
        let token_str = pprust::token_kind_to_string(t);
        let this_token_str = super::token_descr(&self.token);
        let (prev_sp, sp) = match (&self.token.kind, self.subparser_name) {
            // Point at the end of the macro call when reaching end of macro arguments.
            (token::Eof, Some(_)) => {
                let sp = self.prev_token.span.shrink_to_hi();
                (sp, sp)
            }
            // We don't want to point at the following span after DUMMY_SP.
            // This happens when the parser finds an empty TokenStream.
            _ if self.prev_token.span == DUMMY_SP => (self.token.span, self.token.span),
            // EOF, don't want to point at the following char, but rather the last token.
            (token::Eof, None) => (self.prev_token.span, self.token.span),
            _ => (self.prev_token.span.shrink_to_hi(), self.token.span),
        };
        let msg = format!(
            "expected `{}`, found {}",
            token_str,
            match (&self.token.kind, self.subparser_name) {
                (token::Eof, Some(origin)) => format!("end of {origin}"),
                _ => this_token_str,
            },
        );
        let mut err = self.dcx().struct_span_err(sp, msg);
        let label_exp = format!("expected `{token_str}`");
        let sm = self.psess.source_map();
        if !sm.is_multiline(prev_sp.until(sp)) {
            // When the spans are in the same line, it means that the only content
            // between them is whitespace, point only at the found token.
            err.span_label(sp, label_exp);
        } else {
            err.span_label(prev_sp, label_exp);
            err.span_label(sp, "unexpected token");
        }
        Err(err)
    }

    pub(super) fn expect_semi(&mut self) -> PResult<'a, ()> {
        if self.eat(exp!(Semi)) || self.recover_colon_as_semi() {
            return Ok(());
        }
        self.expect(exp!(Semi)).map(drop) // Error unconditionally
    }

    pub(super) fn recover_colon_as_semi(&mut self) -> bool {
        let line_idx = |span: Span| {
            self.psess
                .source_map()
                .span_to_lines(span)
                .ok()
                .and_then(|lines| Some(lines.lines.get(0)?.line_index))
        };

        if self.may_recover()
            && self.token == token::Colon
            && self.look_ahead(1, |next| line_idx(self.token.span) < line_idx(next.span))
        {
            self.dcx().emit_err(ColonAsSemi { span: self.token.span });
            self.bump();
            return true;
        }

        false
    }

    /// Consumes alternative await syntaxes like `await!(<expr>)`, `await <expr>`,
    /// `await? <expr>`, `await(<expr>)`, and `await { <expr> }`.
    pub(super) fn recover_incorrect_await_syntax(
        &mut self,
        await_sp: Span,
    ) -> PResult<'a, P<Expr>> {
        let (hi, expr, is_question) = if self.token == token::Bang {
            // Handle `await!(<expr>)`.
            self.recover_await_macro()?
        } else {
            self.recover_await_prefix(await_sp)?
        };
        let (sp, guar) = self.error_on_incorrect_await(await_sp, hi, &expr, is_question);
        let expr = self.mk_expr_err(await_sp.to(sp), guar);
        self.maybe_recover_from_bad_qpath(expr)
    }

    fn recover_await_macro(&mut self) -> PResult<'a, (Span, P<Expr>, bool)> {
        self.expect(exp!(Bang))?;
        self.expect(exp!(OpenParen))?;
        let expr = self.parse_expr()?;
        self.expect(exp!(CloseParen))?;
        Ok((self.prev_token.span, expr, false))
    }

    fn recover_await_prefix(&mut self, await_sp: Span) -> PResult<'a, (Span, P<Expr>, bool)> {
        let is_question = self.eat(exp!(Question)); // Handle `await? <expr>`.
        let expr = if self.token == token::OpenBrace {
            // Handle `await { <expr> }`.
            // This needs to be handled separately from the next arm to avoid
            // interpreting `await { <expr> }?` as `<expr>?.await`.
            self.parse_expr_block(None, self.token.span, BlockCheckMode::Default)
        } else {
            self.parse_expr()
        }
        .map_err(|mut err| {
            err.span_label(await_sp, format!("while parsing this incorrect await expression"));
            err
        })?;
        Ok((expr.span, expr, is_question))
    }

    fn error_on_incorrect_await(
        &self,
        lo: Span,
        hi: Span,
        expr: &Expr,
        is_question: bool,
    ) -> (Span, ErrorGuaranteed) {
        let span = lo.to(hi);
        let guar = self.dcx().emit_err(IncorrectAwait {
            span,
            suggestion: AwaitSuggestion {
                removal: lo.until(expr.span),
                dot_await: expr.span.shrink_to_hi(),
                question_mark: if is_question { "?" } else { "" },
            },
        });
        (span, guar)
    }

    /// If encountering `future.await()`, consumes and emits an error.
    pub(super) fn recover_from_await_method_call(&mut self) {
        if self.token == token::OpenParen && self.look_ahead(1, |t| t == &token::CloseParen) {
            // future.await()
            let lo = self.token.span;
            self.bump(); // (
            let span = lo.to(self.token.span);
            self.bump(); // )

            self.dcx().emit_err(IncorrectUseOfAwait { span });
        }
    }
    ///
    /// If encountering `x.use()`, consumes and emits an error.
    pub(super) fn recover_from_use(&mut self) {
        if self.token == token::OpenParen && self.look_ahead(1, |t| t == &token::CloseParen) {
            // var.use()
            let lo = self.token.span;
            self.bump(); // (
            let span = lo.to(self.token.span);
            self.bump(); // )

            self.dcx().emit_err(IncorrectUseOfUse { span });
        }
    }

    pub(super) fn try_macro_suggestion(&mut self) -> PResult<'a, P<Expr>> {
        let is_try = self.token.is_keyword(kw::Try);
        let is_questionmark = self.look_ahead(1, |t| t == &token::Bang); //check for !
        let is_open = self.look_ahead(2, |t| t == &token::OpenParen); //check for (

        if is_try && is_questionmark && is_open {
            let lo = self.token.span;
            self.bump(); //remove try
            self.bump(); //remove !
            let try_span = lo.to(self.token.span); //we take the try!( span
            self.bump(); //remove (
            let is_empty = self.token == token::CloseParen; //check if the block is empty
            self.consume_block(exp!(OpenParen), exp!(CloseParen), ConsumeClosingDelim::No); //eat the block
            let hi = self.token.span;
            self.bump(); //remove )
            let mut err = self.dcx().struct_span_err(lo.to(hi), "use of deprecated `try` macro");
            err.note("in the 2018 edition `try` is a reserved keyword, and the `try!()` macro is deprecated");
            let prefix = if is_empty { "" } else { "alternatively, " };
            if !is_empty {
                err.multipart_suggestion(
                    "you can use the `?` operator instead",
                    vec![(try_span, "".to_owned()), (hi, "?".to_owned())],
                    Applicability::MachineApplicable,
                );
            }
            err.span_suggestion(lo.shrink_to_lo(), format!("{prefix}you can still access the deprecated `try!()` macro using the \"raw identifier\" syntax"), "r#", Applicability::MachineApplicable);
            let guar = err.emit();
            Ok(self.mk_expr_err(lo.to(hi), guar))
        } else {
            Err(self.expected_expression_found()) // The user isn't trying to invoke the try! macro
        }
    }

    /// When trying to close a generics list and encountering code like
    /// ```text
    /// impl<S: Into<std::borrow::Cow<'static, str>> From<S> for Canonical {}
    ///                                          // ^ missing > here
    /// ```
    /// we provide a structured suggestion on the error from `expect_gt`.
    pub(super) fn expect_gt_or_maybe_suggest_closing_generics(
        &mut self,
        params: &[ast::GenericParam],
    ) -> PResult<'a, ()> {
        let Err(mut err) = self.expect_gt() else {
            return Ok(());
        };
        // Attempt to find places where a missing `>` might belong.
        if let [.., ast::GenericParam { bounds, .. }] = params
            && let Some(poly) = bounds
                .iter()
                .filter_map(|bound| match bound {
                    ast::GenericBound::Trait(poly) => Some(poly),
                    _ => None,
                })
                .next_back()
        {
            err.span_suggestion_verbose(
                poly.span.shrink_to_hi(),
                "you might have meant to end the type parameters here",
                ">",
                Applicability::MaybeIncorrect,
            );
        }
        Err(err)
    }

    pub(super) fn recover_seq_parse_error(
        &mut self,
        open: ExpTokenPair<'_>,
        close: ExpTokenPair<'_>,
        lo: Span,
        err: Diag<'a>,
    ) -> P<Expr> {
        let guar = err.emit();
        // Recover from parse error, callers expect the closing delim to be consumed.
        self.consume_block(open, close, ConsumeClosingDelim::Yes);
        self.mk_expr(lo.to(self.prev_token.span), ExprKind::Err(guar))
    }

    /// Eats tokens until we can be relatively sure we reached the end of the
    /// statement. This is something of a best-effort heuristic.
    ///
    /// We terminate when we find an unmatched `}` (without consuming it).
    pub(super) fn recover_stmt(&mut self) {
        self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore)
    }

    /// If `break_on_semi` is `Break`, then we will stop consuming tokens after
    /// finding (and consuming) a `;` outside of `{}` or `[]` (note that this is
    /// approximate -- it can mean we break too early due to macros, but that
    /// should only lead to sub-optimal recovery, not inaccurate parsing).
    ///
    /// If `break_on_block` is `Break`, then we will stop consuming tokens
    /// after finding (and consuming) a brace-delimited block.
    pub(super) fn recover_stmt_(
        &mut self,
        break_on_semi: SemiColonMode,
        break_on_block: BlockMode,
    ) {
        let mut brace_depth = 0;
        let mut bracket_depth = 0;
        let mut in_block = false;
        debug!("recover_stmt_ enter loop (semi={:?}, block={:?})", break_on_semi, break_on_block);
        loop {
            debug!("recover_stmt_ loop {:?}", self.token);
            match self.token.kind {
                token::OpenBrace => {
                    brace_depth += 1;
                    self.bump();
                    if break_on_block == BlockMode::Break && brace_depth == 1 && bracket_depth == 0
                    {
                        in_block = true;
                    }
                }
                token::OpenBracket => {
                    bracket_depth += 1;
                    self.bump();
                }
                token::CloseBrace => {
                    if brace_depth == 0 {
                        debug!("recover_stmt_ return - close delim {:?}", self.token);
                        break;
                    }
                    brace_depth -= 1;
                    self.bump();
                    if in_block && bracket_depth == 0 && brace_depth == 0 {
                        debug!("recover_stmt_ return - block end {:?}", self.token);
                        break;
                    }
                }
                token::CloseBracket => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        bracket_depth = 0;
                    }
                    self.bump();
                }
                token::Eof => {
                    debug!("recover_stmt_ return - Eof");
                    break;
                }
                token::Semi => {
                    self.bump();
                    if break_on_semi == SemiColonMode::Break
                        && brace_depth == 0
                        && bracket_depth == 0
                    {
                        debug!("recover_stmt_ return - Semi");
                        break;
                    }
                }
                token::Comma
                    if break_on_semi == SemiColonMode::Comma
                        && brace_depth == 0
                        && bracket_depth == 0 =>
                {
                    break;
                }
                _ => self.bump(),
            }
        }
    }

    pub(super) fn check_for_for_in_in_typo(&mut self, in_span: Span) {
        if self.eat_keyword(exp!(In)) {
            // a common typo: `for _ in in bar {}`
            self.dcx().emit_err(InInTypo {
                span: self.prev_token.span,
                sugg_span: in_span.until(self.prev_token.span),
            });
        }
    }

    pub(super) fn eat_incorrect_doc_comment_for_param_type(&mut self) {
        if let token::DocComment(..) = self.token.kind {
            self.dcx().emit_err(DocCommentOnParamType { span: self.token.span });
            self.bump();
        } else if self.token == token::Pound && self.look_ahead(1, |t| *t == token::OpenBracket) {
            let lo = self.token.span;
            // Skip every token until next possible arg.
            while self.token != token::CloseBracket {
                self.bump();
            }
            let sp = lo.to(self.token.span);
            self.bump();
            self.dcx().emit_err(AttributeOnParamType { span: sp });
        }
    }

    pub(super) fn parameter_without_type(
        &mut self,
        err: &mut Diag<'_>,
        pat: P<ast::Pat>,
        require_name: bool,
        first_param: bool,
    ) -> Option<Ident> {
        // If we find a pattern followed by an identifier, it could be an (incorrect)
        // C-style parameter declaration.
        if self.check_ident()
            && self.look_ahead(1, |t| *t == token::Comma || *t == token::CloseParen)
        {
            // `fn foo(String s) {}`
            let ident = self.parse_ident().unwrap();
            let span = pat.span.with_hi(ident.span.hi());

            err.span_suggestion(
                span,
                "declare the type after the parameter binding",
                "<identifier>: <type>",
                Applicability::HasPlaceholders,
            );
            return Some(ident);
        } else if require_name
            && (self.token == token::Comma
                || self.token == token::Lt
                || self.token == token::CloseParen)
        {
            let rfc_note = "anonymous parameters are removed in the 2018 edition (see RFC 1685)";

            let (ident, self_sugg, param_sugg, type_sugg, self_span, param_span, type_span) =
                match pat.kind {
                    PatKind::Ident(_, ident, _) => (
                        ident,
                        "self: ",
                        ": TypeName".to_string(),
                        "_: ",
                        pat.span.shrink_to_lo(),
                        pat.span.shrink_to_hi(),
                        pat.span.shrink_to_lo(),
                    ),
                    // Also catches `fn foo(&a)`.
                    PatKind::Ref(ref inner_pat, mutab)
                        if let PatKind::Ident(_, ident, _) = inner_pat.clone().kind =>
                    {
                        let mutab = mutab.prefix_str();
                        (
                            ident,
                            "self: ",
                            format!("{ident}: &{mutab}TypeName"),
                            "_: ",
                            pat.span.shrink_to_lo(),
                            pat.span,
                            pat.span.shrink_to_lo(),
                        )
                    }
                    _ => {
                        // Otherwise, try to get a type and emit a suggestion.
                        if let Some(_) = pat.to_ty() {
                            err.span_suggestion_verbose(
                                pat.span.shrink_to_lo(),
                                "explicitly ignore the parameter name",
                                "_: ".to_string(),
                                Applicability::MachineApplicable,
                            );
                            err.note(rfc_note);
                        }

                        return None;
                    }
                };

            // `fn foo(a, b) {}`, `fn foo(a<x>, b<y>) {}` or `fn foo(usize, usize) {}`
            if first_param {
                err.span_suggestion_verbose(
                    self_span,
                    "if this is a `self` type, give it a parameter name",
                    self_sugg,
                    Applicability::MaybeIncorrect,
                );
            }
            // Avoid suggesting that `fn foo(HashMap<u32>)` is fixed with a change to
            // `fn foo(HashMap: TypeName<u32>)`.
            if self.token != token::Lt {
                err.span_suggestion_verbose(
                    param_span,
                    "if this is a parameter name, give it a type",
                    param_sugg,
                    Applicability::HasPlaceholders,
                );
            }
            err.span_suggestion_verbose(
                type_span,
                "if this is a type, explicitly ignore the parameter name",
                type_sugg,
                Applicability::MachineApplicable,
            );
            err.note(rfc_note);

            // Don't attempt to recover by using the `X` in `X<Y>` as the parameter name.
            return if self.token == token::Lt { None } else { Some(ident) };
        }
        None
    }

    pub(super) fn recover_arg_parse(&mut self) -> PResult<'a, (P<ast::Pat>, P<ast::Ty>)> {
        let pat = self.parse_pat_no_top_alt(Some(Expected::ArgumentName), None)?;
        self.expect(exp!(Colon))?;
        let ty = self.parse_ty()?;

        self.dcx().emit_err(PatternMethodParamWithoutBody { span: pat.span });

        // Pretend the pattern is `_`, to avoid duplicate errors from AST validation.
        let pat =
            P(Pat { kind: PatKind::Wild, span: pat.span, id: ast::DUMMY_NODE_ID, tokens: None });
        Ok((pat, ty))
    }

    pub(super) fn recover_bad_self_param(&mut self, mut param: Param) -> PResult<'a, Param> {
        let span = param.pat.span;
        let guar = self.dcx().emit_err(SelfParamNotFirst { span });
        param.ty.kind = TyKind::Err(guar);
        Ok(param)
    }

    pub(super) fn consume_block(
        &mut self,
        open: ExpTokenPair<'_>,
        close: ExpTokenPair<'_>,
        consume_close: ConsumeClosingDelim,
    ) {
        let mut brace_depth = 0;
        loop {
            if self.eat(open) {
                brace_depth += 1;
            } else if self.check(close) {
                if brace_depth == 0 {
                    if let ConsumeClosingDelim::Yes = consume_close {
                        // Some of the callers of this method expect to be able to parse the
                        // closing delimiter themselves, so we leave it alone. Otherwise we advance
                        // the parser.
                        self.bump();
                    }
                    return;
                } else {
                    self.bump();
                    brace_depth -= 1;
                    continue;
                }
            } else if self.token == token::Eof {
                return;
            } else {
                self.bump();
            }
        }
    }

    pub(super) fn expected_expression_found(&self) -> Diag<'a> {
        let (span, msg) = match (&self.token.kind, self.subparser_name) {
            (&token::Eof, Some(origin)) => {
                let sp = self.prev_token.span.shrink_to_hi();
                (sp, format!("expected expression, found end of {origin}"))
            }
            _ => (
                self.token.span,
                format!("expected expression, found {}", super::token_descr(&self.token)),
            ),
        };
        let mut err = self.dcx().struct_span_err(span, msg);
        let sp = self.psess.source_map().start_point(self.token.span);
        if let Some(sp) = self.psess.ambiguous_block_expr_parse.borrow().get(&sp) {
            err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
        }
        err.span_label(span, "expected expression");
        err
    }

    fn consume_tts(
        &mut self,
        mut acc: i64, // `i64` because malformed code can have more closing delims than opening.
        // Not using `FxHashMap` due to `token::TokenKind: !Eq + !Hash`.
        modifier: &[(token::TokenKind, i64)],
    ) {
        while acc > 0 {
            if let Some((_, val)) = modifier.iter().find(|(t, _)| self.token == *t) {
                acc += *val;
            }
            if self.token == token::Eof {
                break;
            }
            self.bump();
        }
    }

    /// Replace duplicated recovered parameters with `_` pattern to avoid unnecessary errors.
    ///
    /// This is necessary because at this point we don't know whether we parsed a function with
    /// anonymous parameters or a function with names but no types. In order to minimize
    /// unnecessary errors, we assume the parameters are in the shape of `fn foo(a, b, c)` where
    /// the parameters are *names* (so we don't emit errors about not being able to find `b` in
    /// the local scope), but if we find the same name multiple times, like in `fn foo(i8, i8)`,
    /// we deduplicate them to not complain about duplicated parameter names.
    pub(super) fn deduplicate_recovered_params_names(&self, fn_inputs: &mut ThinVec<Param>) {
        let mut seen_inputs = FxHashSet::default();
        for input in fn_inputs.iter_mut() {
            let opt_ident = if let (PatKind::Ident(_, ident, _), TyKind::Err(_)) =
                (&input.pat.kind, &input.ty.kind)
            {
                Some(*ident)
            } else {
                None
            };
            if let Some(ident) = opt_ident {
                if seen_inputs.contains(&ident) {
                    input.pat.kind = PatKind::Wild;
                }
                seen_inputs.insert(ident);
            }
        }
    }

    /// Handle encountering a symbol in a generic argument list that is not a `,` or `>`. In this
    /// case, we emit an error and try to suggest enclosing a const argument in braces if it looks
    /// like the user has forgotten them.
    pub(super) fn handle_ambiguous_unbraced_const_arg(
        &mut self,
        args: &mut ThinVec<AngleBracketedArg>,
    ) -> PResult<'a, bool> {
        // If we haven't encountered a closing `>`, then the argument is malformed.
        // It's likely that the user has written a const expression without enclosing it
        // in braces, so we try to recover here.
        let arg = args.pop().unwrap();
        // FIXME: for some reason using `unexpected` or `expected_one_of_not_found` has
        // adverse side-effects to subsequent errors and seems to advance the parser.
        // We are causing this error here exclusively in case that a `const` expression
        // could be recovered from the current parser state, even if followed by more
        // arguments after a comma.
        let mut err = self.dcx().struct_span_err(
            self.token.span,
            format!("expected one of `,` or `>`, found {}", super::token_descr(&self.token)),
        );
        err.span_label(self.token.span, "expected one of `,` or `>`");
        match self.recover_const_arg(arg.span(), err) {
            Ok(arg) => {
                args.push(AngleBracketedArg::Arg(arg));
                if self.eat(exp!(Comma)) {
                    return Ok(true); // Continue
                }
            }
            Err(err) => {
                args.push(arg);
                // We will emit a more generic error later.
                err.delay_as_bug();
            }
        }
        Ok(false) // Don't continue.
    }

    /// Attempt to parse a generic const argument that has not been enclosed in braces.
    /// There are a limited number of expressions that are permitted without being encoded
    /// in braces:
    /// - Literals.
    /// - Single-segment paths (i.e. standalone generic const parameters).
    /// All other expressions that can be parsed will emit an error suggesting the expression be
    /// wrapped in braces.
    pub(super) fn handle_unambiguous_unbraced_const_arg(&mut self) -> PResult<'a, P<Expr>> {
        let start = self.token.span;
        let attrs = self.parse_outer_attributes()?;
        let (expr, _) =
            self.parse_expr_res(Restrictions::CONST_EXPR, attrs).map_err(|mut err| {
                err.span_label(
                    start.shrink_to_lo(),
                    "while parsing a const generic argument starting here",
                );
                err
            })?;
        if !self.expr_is_valid_const_arg(&expr) {
            self.dcx().emit_err(ConstGenericWithoutBraces {
                span: expr.span,
                sugg: ConstGenericWithoutBracesSugg {
                    left: expr.span.shrink_to_lo(),
                    right: expr.span.shrink_to_hi(),
                },
            });
        }
        Ok(expr)
    }

    fn recover_const_param_decl(&mut self, ty_generics: Option<&Generics>) -> Option<GenericArg> {
        let snapshot = self.create_snapshot_for_diagnostic();
        let param = match self.parse_const_param(AttrVec::new()) {
            Ok(param) => param,
            Err(err) => {
                err.cancel();
                self.restore_snapshot(snapshot);
                return None;
            }
        };

        let ident = param.ident.to_string();
        let sugg = match (ty_generics, self.psess.source_map().span_to_snippet(param.span())) {
            (Some(Generics { params, span: impl_generics, .. }), Ok(snippet)) => {
                Some(match &params[..] {
                    [] => UnexpectedConstParamDeclarationSugg::AddParam {
                        impl_generics: *impl_generics,
                        incorrect_decl: param.span(),
                        snippet,
                        ident,
                    },
                    [.., generic] => UnexpectedConstParamDeclarationSugg::AppendParam {
                        impl_generics_end: generic.span().shrink_to_hi(),
                        incorrect_decl: param.span(),
                        snippet,
                        ident,
                    },
                })
            }
            _ => None,
        };
        let guar =
            self.dcx().emit_err(UnexpectedConstParamDeclaration { span: param.span(), sugg });

        let value = self.mk_expr_err(param.span(), guar);
        Some(GenericArg::Const(AnonConst { id: ast::DUMMY_NODE_ID, value }))
    }

    pub(super) fn recover_const_param_declaration(
        &mut self,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Option<GenericArg>> {
        // We have to check for a few different cases.
        if let Some(arg) = self.recover_const_param_decl(ty_generics) {
            return Ok(Some(arg));
        }

        // We haven't consumed `const` yet.
        let start = self.token.span;
        self.bump(); // `const`

        // Detect and recover from the old, pre-RFC2000 syntax for const generics.
        let mut err = UnexpectedConstInGenericParam { span: start, to_remove: None };
        if self.check_const_arg() {
            err.to_remove = Some(start.until(self.token.span));
            self.dcx().emit_err(err);
            Ok(Some(GenericArg::Const(self.parse_const_arg()?)))
        } else {
            let after_kw_const = self.token.span;
            self.recover_const_arg(after_kw_const, self.dcx().create_err(err)).map(Some)
        }
    }

    /// Try to recover from possible generic const argument without `{` and `}`.
    ///
    /// When encountering code like `foo::< bar + 3 >` or `foo::< bar - baz >` we suggest
    /// `foo::<{ bar + 3 }>` and `foo::<{ bar - baz }>`, respectively. We only provide a suggestion
    /// if we think that the resulting expression would be well formed.
    pub(super) fn recover_const_arg(
        &mut self,
        start: Span,
        mut err: Diag<'a>,
    ) -> PResult<'a, GenericArg> {
        let is_op_or_dot = AssocOp::from_token(&self.token)
            .and_then(|op| {
                if let AssocOp::Binary(
                    BinOpKind::Gt
                    | BinOpKind::Lt
                    | BinOpKind::Shr
                    | BinOpKind::Ge
                )
                // Don't recover from `foo::<bar = baz>`, because this could be an attempt to
                // assign a value to a defaulted generic parameter.
                | AssocOp::Assign
                | AssocOp::AssignOp(_) = op
                {
                    None
                } else {
                    Some(op)
                }
            })
            .is_some()
            || self.token == TokenKind::Dot;
        // This will be true when a trait object type `Foo +` or a path which was a `const fn` with
        // type params has been parsed.
        let was_op = matches!(self.prev_token.kind, token::Plus | token::Shr | token::Gt);
        if !is_op_or_dot && !was_op {
            // We perform these checks and early return to avoid taking a snapshot unnecessarily.
            return Err(err);
        }
        let snapshot = self.create_snapshot_for_diagnostic();
        if is_op_or_dot {
            self.bump();
        }
        match (|| {
            let attrs = self.parse_outer_attributes()?;
            self.parse_expr_res(Restrictions::CONST_EXPR, attrs)
        })() {
            Ok((expr, _)) => {
                // Find a mistake like `MyTrait<Assoc == S::Assoc>`.
                if snapshot.token == token::EqEq {
                    err.span_suggestion(
                        snapshot.token.span,
                        "if you meant to use an associated type binding, replace `==` with `=`",
                        "=",
                        Applicability::MaybeIncorrect,
                    );
                    let guar = err.emit();
                    let value = self.mk_expr_err(start.to(expr.span), guar);
                    return Ok(GenericArg::Const(AnonConst { id: ast::DUMMY_NODE_ID, value }));
                } else if snapshot.token == token::Colon
                    && expr.span.lo() == snapshot.token.span.hi()
                    && matches!(expr.kind, ExprKind::Path(..))
                {
                    // Find a mistake like "foo::var:A".
                    err.span_suggestion(
                        snapshot.token.span,
                        "write a path separator here",
                        "::",
                        Applicability::MaybeIncorrect,
                    );
                    let guar = err.emit();
                    return Ok(GenericArg::Type(
                        self.mk_ty(start.to(expr.span), TyKind::Err(guar)),
                    ));
                } else if self.token == token::Comma || self.token.kind.should_end_const_arg() {
                    // Avoid the following output by checking that we consumed a full const arg:
                    // help: expressions must be enclosed in braces to be used as const generic
                    //       arguments
                    //    |
                    // LL |     let sr: Vec<{ (u32, _, _) = vec![] };
                    //    |                 ^                      ^
                    return Ok(self.dummy_const_arg_needs_braces(err, start.to(expr.span)));
                }
            }
            Err(err) => {
                err.cancel();
            }
        }
        self.restore_snapshot(snapshot);
        Err(err)
    }

    /// Try to recover from an unbraced const argument whose first token [could begin a type][ty].
    ///
    /// [ty]: token::Token::can_begin_type
    pub(crate) fn recover_unbraced_const_arg_that_can_begin_ty(
        &mut self,
        mut snapshot: SnapshotParser<'a>,
    ) -> Option<P<ast::Expr>> {
        match (|| {
            let attrs = self.parse_outer_attributes()?;
            snapshot.parse_expr_res(Restrictions::CONST_EXPR, attrs)
        })() {
            // Since we don't know the exact reason why we failed to parse the type or the
            // expression, employ a simple heuristic to weed out some pathological cases.
            Ok((expr, _)) if let token::Comma | token::Gt = snapshot.token.kind => {
                self.restore_snapshot(snapshot);
                Some(expr)
            }
            Ok(_) => None,
            Err(err) => {
                err.cancel();
                None
            }
        }
    }

    /// Creates a dummy const argument, and reports that the expression must be enclosed in braces
    pub(super) fn dummy_const_arg_needs_braces(&self, mut err: Diag<'a>, span: Span) -> GenericArg {
        err.multipart_suggestion(
            "expressions must be enclosed in braces to be used as const generic \
             arguments",
            vec![(span.shrink_to_lo(), "{ ".to_string()), (span.shrink_to_hi(), " }".to_string())],
            Applicability::MaybeIncorrect,
        );
        let guar = err.emit();
        let value = self.mk_expr_err(span, guar);
        GenericArg::Const(AnonConst { id: ast::DUMMY_NODE_ID, value })
    }

    /// Some special error handling for the "top-level" patterns in a match arm,
    /// `for` loop, `let`, &c. (in contrast to subpatterns within such).
    pub(crate) fn maybe_recover_colon_colon_in_pat_typo(
        &mut self,
        mut first_pat: P<Pat>,
        expected: Option<Expected>,
    ) -> P<Pat> {
        if token::Colon != self.token.kind {
            return first_pat;
        }
        if !matches!(first_pat.kind, PatKind::Ident(_, _, None) | PatKind::Path(..))
            || !self.look_ahead(1, |token| token.is_non_reserved_ident())
        {
            let mut snapshot_type = self.create_snapshot_for_diagnostic();
            snapshot_type.bump(); // `:`
            match snapshot_type.parse_ty() {
                Err(inner_err) => {
                    inner_err.cancel();
                }
                Ok(ty) => {
                    let Err(mut err) = self.expected_one_of_not_found(&[], &[]) else {
                        return first_pat;
                    };
                    err.span_label(ty.span, "specifying the type of a pattern isn't supported");
                    self.restore_snapshot(snapshot_type);
                    let span = first_pat.span.to(ty.span);
                    first_pat = self.mk_pat(span, PatKind::Wild);
                    err.emit();
                }
            }
            return first_pat;
        }
        // The pattern looks like it might be a path with a `::` -> `:` typo:
        // `match foo { bar:baz => {} }`
        let colon_span = self.token.span;
        // We only emit "unexpected `:`" error here if we can successfully parse the
        // whole pattern correctly in that case.
        let mut snapshot_pat = self.create_snapshot_for_diagnostic();
        let mut snapshot_type = self.create_snapshot_for_diagnostic();

        // Create error for "unexpected `:`".
        match self.expected_one_of_not_found(&[], &[]) {
            Err(mut err) => {
                // Skip the `:`.
                snapshot_pat.bump();
                snapshot_type.bump();
                match snapshot_pat.parse_pat_no_top_alt(expected, None) {
                    Err(inner_err) => {
                        inner_err.cancel();
                    }
                    Ok(mut pat) => {
                        // We've parsed the rest of the pattern.
                        let new_span = first_pat.span.to(pat.span);
                        let mut show_sugg = false;
                        // Try to construct a recovered pattern.
                        match &mut pat.kind {
                            PatKind::Struct(qself @ None, path, ..)
                            | PatKind::TupleStruct(qself @ None, path, _)
                            | PatKind::Path(qself @ None, path) => match &first_pat.kind {
                                PatKind::Ident(_, ident, _) => {
                                    path.segments.insert(0, PathSegment::from_ident(*ident));
                                    path.span = new_span;
                                    show_sugg = true;
                                    first_pat = pat;
                                }
                                PatKind::Path(old_qself, old_path) => {
                                    path.segments = old_path
                                        .segments
                                        .iter()
                                        .cloned()
                                        .chain(take(&mut path.segments))
                                        .collect();
                                    path.span = new_span;
                                    *qself = old_qself.clone();
                                    first_pat = pat;
                                    show_sugg = true;
                                }
                                _ => {}
                            },
                            PatKind::Ident(BindingMode::NONE, ident, None) => {
                                match &first_pat.kind {
                                    PatKind::Ident(_, old_ident, _) => {
                                        let path = PatKind::Path(
                                            None,
                                            Path {
                                                span: new_span,
                                                segments: thin_vec![
                                                    PathSegment::from_ident(*old_ident),
                                                    PathSegment::from_ident(*ident),
                                                ],
                                                tokens: None,
                                            },
                                        );
                                        first_pat = self.mk_pat(new_span, path);
                                        show_sugg = true;
                                    }
                                    PatKind::Path(old_qself, old_path) => {
                                        let mut segments = old_path.segments.clone();
                                        segments.push(PathSegment::from_ident(*ident));
                                        let path = PatKind::Path(
                                            old_qself.clone(),
                                            Path { span: new_span, segments, tokens: None },
                                        );
                                        first_pat = self.mk_pat(new_span, path);
                                        show_sugg = true;
                                    }
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                        if show_sugg {
                            err.span_suggestion_verbose(
                                colon_span.until(self.look_ahead(1, |t| t.span)),
                                "maybe write a path separator here",
                                "::",
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            first_pat = self.mk_pat(new_span, PatKind::Wild);
                        }
                        self.restore_snapshot(snapshot_pat);
                    }
                }
                match snapshot_type.parse_ty() {
                    Err(inner_err) => {
                        inner_err.cancel();
                    }
                    Ok(ty) => {
                        err.span_label(ty.span, "specifying the type of a pattern isn't supported");
                        self.restore_snapshot(snapshot_type);
                        let new_span = first_pat.span.to(ty.span);
                        first_pat = self.mk_pat(new_span, PatKind::Wild);
                    }
                }
                err.emit();
            }
            _ => {
                // Carry on as if we had not done anything. This should be unreachable.
            }
        };
        first_pat
    }

    /// If `loop_header` is `Some` and an unexpected block label is encountered,
    /// it is suggested to be moved just before `loop_header`, else it is suggested to be removed.
    pub(crate) fn maybe_recover_unexpected_block_label(
        &mut self,
        loop_header: Option<Span>,
    ) -> bool {
        // Check for `'a : {`
        if !(self.check_lifetime()
            && self.look_ahead(1, |t| *t == token::Colon)
            && self.look_ahead(2, |t| *t == token::OpenBrace))
        {
            return false;
        }
        let label = self.eat_label().expect("just checked if a label exists");
        self.bump(); // eat `:`
        let span = label.ident.span.to(self.prev_token.span);
        let mut diag = self
            .dcx()
            .struct_span_err(span, "block label not supported here")
            .with_span_label(span, "not supported here");
        if let Some(loop_header) = loop_header {
            diag.multipart_suggestion(
                "if you meant to label the loop, move this label before the loop",
                vec![
                    (label.ident.span.until(self.token.span), String::from("")),
                    (loop_header.shrink_to_lo(), format!("{}: ", label.ident)),
                ],
                Applicability::MachineApplicable,
            );
        } else {
            diag.tool_only_span_suggestion(
                label.ident.span.until(self.token.span),
                "remove this block label",
                "",
                Applicability::MachineApplicable,
            );
        }
        diag.emit();
        true
    }

    /// Some special error handling for the "top-level" patterns in a match arm,
    /// `for` loop, `let`, &c. (in contrast to subpatterns within such).
    pub(crate) fn maybe_recover_unexpected_comma(
        &mut self,
        lo: Span,
        rt: CommaRecoveryMode,
    ) -> PResult<'a, ()> {
        if self.token != token::Comma {
            return Ok(());
        }

        // An unexpected comma after a top-level pattern is a clue that the
        // user (perhaps more accustomed to some other language) forgot the
        // parentheses in what should have been a tuple pattern; return a
        // suggestion-enhanced error here rather than choking on the comma later.
        let comma_span = self.token.span;
        self.bump();
        if let Err(err) = self.skip_pat_list() {
            // We didn't expect this to work anyway; we just wanted to advance to the
            // end of the comma-sequence so we know the span to suggest parenthesizing.
            err.cancel();
        }
        let seq_span = lo.to(self.prev_token.span);
        let mut err = self.dcx().struct_span_err(comma_span, "unexpected `,` in pattern");
        if let Ok(seq_snippet) = self.span_to_snippet(seq_span) {
            err.multipart_suggestion(
                format!(
                    "try adding parentheses to match on a tuple{}",
                    if let CommaRecoveryMode::LikelyTuple = rt { "" } else { "..." },
                ),
                vec![
                    (seq_span.shrink_to_lo(), "(".to_string()),
                    (seq_span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MachineApplicable,
            );
            if let CommaRecoveryMode::EitherTupleOrPipe = rt {
                err.span_suggestion(
                    seq_span,
                    "...or a vertical bar to match on multiple alternatives",
                    seq_snippet.replace(',', " |"),
                    Applicability::MachineApplicable,
                );
            }
        }
        Err(err)
    }

    pub(crate) fn maybe_recover_bounds_doubled_colon(&mut self, ty: &Ty) -> PResult<'a, ()> {
        let TyKind::Path(qself, path) = &ty.kind else { return Ok(()) };
        let qself_position = qself.as_ref().map(|qself| qself.position);
        for (i, segments) in path.segments.windows(2).enumerate() {
            if qself_position.is_some_and(|pos| i < pos) {
                continue;
            }
            if let [a, b] = segments {
                let (a_span, b_span) = (a.span(), b.span());
                let between_span = a_span.shrink_to_hi().to(b_span.shrink_to_lo());
                if self.span_to_snippet(between_span).as_deref() == Ok(":: ") {
                    return Err(self.dcx().create_err(DoubleColonInBound {
                        span: path.span.shrink_to_hi(),
                        between: between_span,
                    }));
                }
            }
        }
        Ok(())
    }

    /// Check for exclusive ranges written as `..<`
    pub(crate) fn maybe_err_dotdotlt_syntax(&self, maybe_lt: Token, mut err: Diag<'a>) -> Diag<'a> {
        if maybe_lt == token::Lt
            && (self.expected_token_types.contains(TokenType::Gt)
                || matches!(self.token.kind, token::Literal(..)))
        {
            err.span_suggestion(
                maybe_lt.span,
                "remove the `<` to write an exclusive range",
                "",
                Applicability::MachineApplicable,
            );
        }
        err
    }

    /// This checks if this is a conflict marker, depending of the parameter passed.
    ///
    /// * `<<<<<<<`
    /// * `|||||||`
    /// * `=======`
    /// * `>>>>>>>`
    ///
    pub(super) fn is_vcs_conflict_marker(
        &mut self,
        long_kind: &TokenKind,
        short_kind: &TokenKind,
    ) -> bool {
        (0..3).all(|i| self.look_ahead(i, |tok| tok == long_kind))
            && self.look_ahead(3, |tok| tok == short_kind)
    }

    fn conflict_marker(&mut self, long_kind: &TokenKind, short_kind: &TokenKind) -> Option<Span> {
        if self.is_vcs_conflict_marker(long_kind, short_kind) {
            let lo = self.token.span;
            for _ in 0..4 {
                self.bump();
            }
            return Some(lo.to(self.prev_token.span));
        }
        None
    }

    pub(super) fn recover_vcs_conflict_marker(&mut self) {
        // <<<<<<<
        let Some(start) = self.conflict_marker(&TokenKind::Shl, &TokenKind::Lt) else {
            return;
        };
        let mut spans = Vec::with_capacity(3);
        spans.push(start);
        // |||||||
        let mut middlediff3 = None;
        // =======
        let mut middle = None;
        // >>>>>>>
        let mut end = None;
        loop {
            if self.token == TokenKind::Eof {
                break;
            }
            if let Some(span) = self.conflict_marker(&TokenKind::OrOr, &TokenKind::Or) {
                middlediff3 = Some(span);
            }
            if let Some(span) = self.conflict_marker(&TokenKind::EqEq, &TokenKind::Eq) {
                middle = Some(span);
            }
            if let Some(span) = self.conflict_marker(&TokenKind::Shr, &TokenKind::Gt) {
                spans.push(span);
                end = Some(span);
                break;
            }
            self.bump();
        }

        let mut err = self.dcx().struct_span_fatal(spans, "encountered diff marker");
        match middlediff3 {
            // We're using diff3
            Some(middlediff3) => {
                err.span_label(
                    start,
                    "between this marker and `|||||||` is the code that we're merging into",
                );
                err.span_label(middlediff3, "between this marker and `=======` is the base code (what the two refs diverged from)");
            }
            None => {
                err.span_label(
                    start,
                    "between this marker and `=======` is the code that we're merging into",
                );
            }
        };

        if let Some(middle) = middle {
            err.span_label(middle, "between this marker and `>>>>>>>` is the incoming code");
        }
        if let Some(end) = end {
            err.span_label(end, "this marker concludes the conflict region");
        }
        err.note(
            "conflict markers indicate that a merge was started but could not be completed due \
             to merge conflicts\n\
             to resolve a conflict, keep only the code you want and then delete the lines \
             containing conflict markers",
        );
        err.help(
            "if you're having merge conflicts after pulling new code:\n\
             the top section is the code you already had and the bottom section is the remote code\n\
             if you're in the middle of a rebase:\n\
             the top section is the code being rebased onto and the bottom section is the code \
             coming from the current commit being rebased",
        );

        err.note(
            "for an explanation on these markers from the `git` documentation:\n\
             visit <https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging#_checking_out_conflicts>",
        );

        err.emit();
    }

    /// Parse and throw away a parenthesized comma separated
    /// sequence of patterns until `)` is reached.
    fn skip_pat_list(&mut self) -> PResult<'a, ()> {
        while !self.check(exp!(CloseParen)) {
            self.parse_pat_no_top_alt(None, None)?;
            if !self.eat(exp!(Comma)) {
                return Ok(());
            }
        }
        Ok(())
    }
}
