use super::pat::Expected;
use super::ty::{AllowPlus, RecoverQuestionMark};
use super::{
    BlockMode, Parser, PathStyle, RecoverColon, RecoverComma, Restrictions, SemiColonMode, SeqSep,
    TokenExpectType, TokenType,
};

use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Lit, LitKind, TokenKind};
use rustc_ast::util::parser::AssocOp;
use rustc_ast::{
    AngleBracketedArg, AngleBracketedArgs, AnonConst, AttrVec, BinOpKind, BindingMode, Block,
    BlockCheckMode, Expr, ExprKind, GenericArg, Generics, Item, ItemKind, Mutability, Param, Pat,
    PatKind, Path, PathSegment, QSelf, Ty, TyKind,
};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{pluralize, struct_span_err};
use rustc_errors::{Applicability, DiagnosticBuilder, Handler, PResult};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{MultiSpan, Span, SpanSnippetError, DUMMY_SP};

use std::mem::take;

use tracing::{debug, trace};

const TURBOFISH_SUGGESTION_STR: &str =
    "use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments";

/// Creates a placeholder argument.
pub(super) fn dummy_arg(ident: Ident) -> Param {
    let pat = P(Pat {
        id: ast::DUMMY_NODE_ID,
        kind: PatKind::Ident(BindingMode::ByValue(Mutability::Not), ident, None),
        span: ident.span,
        tokens: None,
    });
    let ty = Ty { kind: TyKind::Err, span: ident.span, id: ast::DUMMY_NODE_ID, tokens: None };
    Param {
        attrs: AttrVec::default(),
        id: ast::DUMMY_NODE_ID,
        pat,
        span: ident.span,
        ty: P(ty),
        is_placeholder: false,
    }
}

pub enum Error {
    UselessDocComment,
}

impl Error {
    fn span_err(self, sp: impl Into<MultiSpan>, handler: &Handler) -> DiagnosticBuilder<'_> {
        match self {
            Error::UselessDocComment => {
                let mut err = struct_span_err!(
                    handler,
                    sp,
                    E0585,
                    "found a documentation comment that doesn't document anything",
                );
                err.help(
                    "doc comments must come before what they document, maybe a comment was \
                          intended with `//`?",
                );
                err
            }
        }
    }
}

pub(super) trait RecoverQPath: Sized + 'static {
    const PATH_STYLE: PathStyle = PathStyle::Expr;
    fn to_ty(&self) -> Option<P<Ty>>;
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self;
}

impl RecoverQPath for Ty {
    const PATH_STYLE: PathStyle = PathStyle::Type;
    fn to_ty(&self) -> Option<P<Ty>> {
        Some(P(self.clone()))
    }
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            kind: TyKind::Path(qself, path),
            id: ast::DUMMY_NODE_ID,
            tokens: None,
        }
    }
}

impl RecoverQPath for Pat {
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self {
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
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self {
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
crate enum ConsumeClosingDelim {
    Yes,
    No,
}

#[derive(Clone, Copy)]
pub enum AttemptLocalParseRecovery {
    Yes,
    No,
}

impl AttemptLocalParseRecovery {
    pub fn yes(&self) -> bool {
        match self {
            AttemptLocalParseRecovery::Yes => true,
            AttemptLocalParseRecovery::No => false,
        }
    }

    pub fn no(&self) -> bool {
        match self {
            AttemptLocalParseRecovery::Yes => false,
            AttemptLocalParseRecovery::No => true,
        }
    }
}

impl<'a> Parser<'a> {
    pub(super) fn span_err<S: Into<MultiSpan>>(&self, sp: S, err: Error) -> DiagnosticBuilder<'a> {
        err.span_err(sp, self.diagnostic())
    }

    pub fn struct_span_err<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_err(sp, m)
    }

    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(sp, m)
    }

    pub(super) fn diagnostic(&self) -> &'a Handler {
        &self.sess.span_diagnostic
    }

    pub(super) fn span_to_snippet(&self, span: Span) -> Result<String, SpanSnippetError> {
        self.sess.source_map().span_to_snippet(span)
    }

    pub(super) fn expected_ident_found(&self) -> DiagnosticBuilder<'a> {
        let mut err = self.struct_span_err(
            self.token.span,
            &format!("expected identifier, found {}", super::token_descr(&self.token)),
        );
        let valid_follow = &[
            TokenKind::Eq,
            TokenKind::Colon,
            TokenKind::Comma,
            TokenKind::Semi,
            TokenKind::ModSep,
            TokenKind::OpenDelim(token::DelimToken::Brace),
            TokenKind::OpenDelim(token::DelimToken::Paren),
            TokenKind::CloseDelim(token::DelimToken::Brace),
            TokenKind::CloseDelim(token::DelimToken::Paren),
        ];
        match self.token.ident() {
            Some((ident, false))
                if ident.is_raw_guess()
                    && self.look_ahead(1, |t| valid_follow.contains(&t.kind)) =>
            {
                err.span_suggestion_verbose(
                    ident.span.shrink_to_lo(),
                    &format!("escape `{}` to use it as an identifier", ident.name),
                    "r#".to_owned(),
                    Applicability::MaybeIncorrect,
                );
            }
            _ => {}
        }
        if let Some(token_descr) = super::token_descr_opt(&self.token) {
            err.span_label(self.token.span, format!("expected identifier, found {}", token_descr));
        } else {
            err.span_label(self.token.span, "expected identifier");
            if self.token == token::Comma && self.look_ahead(1, |t| t.is_ident()) {
                err.span_suggestion(
                    self.token.span,
                    "remove this comma",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
        err
    }

    pub(super) fn expected_one_of_not_found(
        &mut self,
        edible: &[TokenKind],
        inedible: &[TokenKind],
    ) -> PResult<'a, bool /* recovered */> {
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

        let mut expected = edible
            .iter()
            .map(|x| TokenType::Token(x.clone()))
            .chain(inedible.iter().map(|x| TokenType::Token(x.clone())))
            .chain(self.expected_tokens.iter().cloned())
            .collect::<Vec<_>>();
        expected.sort_by_cached_key(|x| x.to_string());
        expected.dedup();

        let sm = self.sess.source_map();
        let msg = format!("expected `;`, found {}", super::token_descr(&self.token));
        let appl = Applicability::MachineApplicable;
        if expected.contains(&TokenType::Token(token::Semi)) {
            if self.token.span == DUMMY_SP || self.prev_token.span == DUMMY_SP {
                // Likely inside a macro, can't provide meaningful suggestions.
            } else if !sm.is_multiline(self.prev_token.span.until(self.token.span)) {
                // The current token is in the same line as the prior token, not recoverable.
            } else if [token::Comma, token::Colon].contains(&self.token.kind)
                && self.prev_token.kind == token::CloseDelim(token::Paren)
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
                t == &token::CloseDelim(token::Brace)
                    || t.can_begin_expr() && t.kind != token::Colon
            }) && [token::Comma, token::Colon].contains(&self.token.kind)
            {
                // Likely typo: `,` → `;` or `:` → `;`. This is triggered if the current token is
                // either `,` or `:`, and the next token could either start a new statement or is a
                // block close. For example:
                //
                //   let x = 32:
                //   let y = 42;
                self.bump();
                let sp = self.prev_token.span;
                self.struct_span_err(sp, &msg)
                    .span_suggestion_short(sp, "change this to `;`", ";".to_string(), appl)
                    .emit();
                return Ok(true);
            } else if self.look_ahead(0, |t| {
                t == &token::CloseDelim(token::Brace)
                    || (
                        t.can_begin_expr() && t != &token::Semi && t != &token::Pound
                        // Avoid triggering with too many trailing `#` in raw string.
                    )
            }) {
                // Missing semicolon typo. This is triggered if the next token could either start a
                // new statement or is a block close. For example:
                //
                //   let x = 32
                //   let y = 42;
                let sp = self.prev_token.span.shrink_to_hi();
                self.struct_span_err(sp, &msg)
                    .span_label(self.token.span, "unexpected token")
                    .span_suggestion_short(sp, "add `;` here", ";".to_string(), appl)
                    .emit();
                return Ok(true);
            }
        }

        let expect = tokens_to_string(&expected);
        let actual = super::token_descr(&self.token);
        let (msg_exp, (label_sp, label_exp)) = if expected.len() > 1 {
            let short_expect = if expected.len() > 6 {
                format!("{} possible tokens", expected.len())
            } else {
                expect.clone()
            };
            (
                format!("expected one of {}, found {}", expect, actual),
                (self.prev_token.span.shrink_to_hi(), format!("expected one of {}", short_expect)),
            )
        } else if expected.is_empty() {
            (
                format!("unexpected token: {}", actual),
                (self.prev_token.span, "unexpected token after this".to_string()),
            )
        } else {
            (
                format!("expected {}, found {}", expect, actual),
                (self.prev_token.span.shrink_to_hi(), format!("expected {}", expect)),
            )
        };
        self.last_unexpected_token_span = Some(self.token.span);
        let mut err = self.struct_span_err(self.token.span, &msg_exp);

        // Add suggestion for a missing closing angle bracket if '>' is included in expected_tokens
        // there are unclosed angle brackets
        if self.unmatched_angle_bracket_count > 0
            && self.token.kind == TokenKind::Eq
            && expected.iter().any(|tok| matches!(tok, TokenType::Token(TokenKind::Gt)))
        {
            err.span_label(self.prev_token.span, "maybe try to close unmatched angle bracket");
        }

        let sp = if self.token == token::Eof {
            // This is EOF; don't want to point at the following char, but rather the last token.
            self.prev_token.span
        } else {
            label_sp
        };
        match self.recover_closing_delimiter(
            &expected
                .iter()
                .filter_map(|tt| match tt {
                    TokenType::Token(t) => Some(t.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            err,
        ) {
            Err(e) => err = e,
            Ok(recovered) => {
                return Ok(recovered);
            }
        }

        if self.check_too_many_raw_str_terminators(&mut err) {
            return Err(err);
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
        self.maybe_annotate_with_ascription(&mut err, false);
        Err(err)
    }

    fn check_too_many_raw_str_terminators(&mut self, err: &mut DiagnosticBuilder<'_>) -> bool {
        match (&self.prev_token.kind, &self.token.kind) {
            (
                TokenKind::Literal(Lit {
                    kind: LitKind::StrRaw(n_hashes) | LitKind::ByteStrRaw(n_hashes),
                    ..
                }),
                TokenKind::Pound,
            ) => {
                err.set_primary_message("too many `#` when terminating raw string");
                err.span_suggestion(
                    self.token.span,
                    "remove the extra `#`",
                    String::new(),
                    Applicability::MachineApplicable,
                );
                err.note(&format!("the raw string started with {} `#`s", n_hashes));
                true
            }
            _ => false,
        }
    }

    pub fn maybe_suggest_struct_literal(
        &mut self,
        lo: Span,
        s: BlockCheckMode,
    ) -> Option<PResult<'a, P<Block>>> {
        if self.token.is_ident() && self.look_ahead(1, |t| t == &token::Colon) {
            // We might be having a struct literal where people forgot to include the path:
            // fn foo() -> Foo {
            //     field: value,
            // }
            let mut snapshot = self.clone();
            let path =
                Path { segments: vec![], span: self.prev_token.span.shrink_to_lo(), tokens: None };
            let struct_expr = snapshot.parse_struct_expr(None, path, AttrVec::new(), false);
            let block_tail = self.parse_block_tail(lo, s, AttemptLocalParseRecovery::No);
            return Some(match (struct_expr, block_tail) {
                (Ok(expr), Err(mut err)) => {
                    // We have encountered the following:
                    // fn foo() -> Foo {
                    //     field: value,
                    // }
                    // Suggest:
                    // fn foo() -> Foo { Path {
                    //     field: value,
                    // } }
                    err.delay_as_bug();
                    self.struct_span_err(expr.span, "struct literal body without path")
                        .multipart_suggestion(
                            "you might have forgotten to add the struct literal inside the block",
                            vec![
                                (expr.span.shrink_to_lo(), "{ SomeStruct ".to_string()),
                                (expr.span.shrink_to_hi(), " }".to_string()),
                            ],
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                    *self = snapshot;
                    let mut tail = self.mk_block(
                        vec![self.mk_stmt_err(expr.span)],
                        s,
                        lo.to(self.prev_token.span),
                    );
                    tail.could_be_bare_literal = true;
                    Ok(tail)
                }
                (Err(mut err), Ok(tail)) => {
                    // We have a block tail that contains a somehow valid type ascription expr.
                    err.cancel();
                    Ok(tail)
                }
                (Err(mut snapshot_err), Err(err)) => {
                    // We don't know what went wrong, emit the normal error.
                    snapshot_err.cancel();
                    self.consume_block(token::Brace, ConsumeClosingDelim::Yes);
                    Err(err)
                }
                (Ok(_), Ok(mut tail)) => {
                    tail.could_be_bare_literal = true;
                    Ok(tail)
                }
            });
        }
        None
    }

    pub fn maybe_annotate_with_ascription(
        &mut self,
        err: &mut DiagnosticBuilder<'_>,
        maybe_expected_semicolon: bool,
    ) {
        if let Some((sp, likely_path)) = self.last_type_ascription.take() {
            let sm = self.sess.source_map();
            let next_pos = sm.lookup_char_pos(self.token.span.lo());
            let op_pos = sm.lookup_char_pos(sp.hi());

            let allow_unstable = self.sess.unstable_features.is_nightly_build();

            if likely_path {
                err.span_suggestion(
                    sp,
                    "maybe write a path separator here",
                    "::".to_string(),
                    if allow_unstable {
                        Applicability::MaybeIncorrect
                    } else {
                        Applicability::MachineApplicable
                    },
                );
                self.sess.type_ascription_path_suggestions.borrow_mut().insert(sp);
            } else if op_pos.line != next_pos.line && maybe_expected_semicolon {
                err.span_suggestion(
                    sp,
                    "try using a semicolon",
                    ";".to_string(),
                    Applicability::MaybeIncorrect,
                );
            } else if allow_unstable {
                err.span_label(sp, "tried to parse a type due to this type ascription");
            } else {
                err.span_label(sp, "tried to parse a type due to this");
            }
            if allow_unstable {
                // Give extra information about type ascription only if it's a nightly compiler.
                err.note(
                    "`#![feature(type_ascription)]` lets you annotate an expression with a type: \
                     `<expr>: <type>`",
                );
                if !likely_path {
                    // Avoid giving too much info when it was likely an unrelated typo.
                    err.note(
                        "see issue #23416 <https://github.com/rust-lang/rust/issues/23416> \
                        for more information",
                    );
                }
            }
        }
    }

    /// Eats and discards tokens until one of `kets` is encountered. Respects token trees,
    /// passes through any errors encountered. Used for error recovery.
    pub(super) fn eat_to_tokens(&mut self, kets: &[&TokenKind]) {
        if let Err(ref mut err) =
            self.parse_seq_to_before_tokens(kets, SeqSep::none(), TokenExpectType::Expect, |p| {
                Ok(p.parse_token_tree())
            })
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
        end: &[&TokenKind],
    ) -> bool {
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
            segment.args.as_ref().map_or(false, |args| args.is_angle_bracketed());

        debug!(
            "check_trailing_angle_brackets: parsed_angle_bracket_args={:?}",
            parsed_angle_bracket_args,
        );
        if !parsed_angle_bracket_args {
            return false;
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
            if *t == token::BinOp(token::BinOpToken::Shr) {
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
            return false;
        }

        // Finally, double check that we have our end token as otherwise this is the
        // second case.
        if self.look_ahead(position, |t| {
            trace!("check_trailing_angle_brackets: t={:?}", t);
            end.contains(&&t.kind)
        }) {
            // Eat from where we started until the end token so that parsing can continue
            // as if we didn't have those extra angle brackets.
            self.eat_to_tokens(end);
            let span = lo.until(self.token.span);

            let total_num_of_gt = number_of_gt + number_of_shr * 2;
            self.struct_span_err(
                span,
                &format!("unmatched angle bracket{}", pluralize!(total_num_of_gt)),
            )
            .span_suggestion(
                span,
                &format!("remove extra angle bracket{}", pluralize!(total_num_of_gt)),
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();
            return true;
        }
        false
    }

    /// Check if a method call with an intended turbofish has been written without surrounding
    /// angle brackets.
    pub(super) fn check_turbofish_missing_angle_brackets(&mut self, segment: &mut PathSegment) {
        if token::ModSep == self.token.kind && segment.args.is_none() {
            let snapshot = self.clone();
            self.bump();
            let lo = self.token.span;
            match self.parse_angle_args(None) {
                Ok(args) => {
                    let span = lo.to(self.prev_token.span);
                    // Detect trailing `>` like in `x.collect::Vec<_>>()`.
                    let mut trailing_span = self.prev_token.span.shrink_to_hi();
                    while self.token.kind == token::BinOp(token::Shr)
                        || self.token.kind == token::Gt
                    {
                        trailing_span = trailing_span.to(self.token.span);
                        self.bump();
                    }
                    if self.token.kind == token::OpenDelim(token::Paren) {
                        // Recover from bad turbofish: `foo.collect::Vec<_>()`.
                        let args = AngleBracketedArgs { args, span }.into();
                        segment.args = args;

                        self.struct_span_err(
                            span,
                            "generic parameters without surrounding angle brackets",
                        )
                        .multipart_suggestion(
                            "surround the type parameters with angle brackets",
                            vec![
                                (span.shrink_to_lo(), "<".to_string()),
                                (trailing_span, ">".to_string()),
                            ],
                            Applicability::MachineApplicable,
                        )
                        .emit();
                    } else {
                        // This doesn't look like an invalid turbofish, can't recover parse state.
                        *self = snapshot;
                    }
                }
                Err(mut err) => {
                    // We couldn't parse generic parameters, unlikely to be a turbofish. Rely on
                    // generic parse error instead.
                    err.cancel();
                    *self = snapshot;
                }
            }
        }
    }

    /// When writing a turbofish with multiple type parameters missing the leading `::`, we will
    /// encounter a parse error when encountering the first `,`.
    pub(super) fn check_mistyped_turbofish_with_multiple_type_params(
        &mut self,
        mut e: DiagnosticBuilder<'a>,
        expr: &mut P<Expr>,
    ) -> PResult<'a, ()> {
        if let ExprKind::Binary(binop, _, _) = &expr.kind {
            if let ast::BinOpKind::Lt = binop.node {
                if self.eat(&token::Comma) {
                    let x = self.parse_seq_to_before_end(
                        &token::Gt,
                        SeqSep::trailing_allowed(token::Comma),
                        |p| p.parse_generic_arg(None),
                    );
                    match x {
                        Ok((_, _, false)) => {
                            if self.eat(&token::Gt) {
                                e.span_suggestion_verbose(
                                    binop.span.shrink_to_lo(),
                                    TURBOFISH_SUGGESTION_STR,
                                    "::".to_string(),
                                    Applicability::MaybeIncorrect,
                                )
                                .emit();
                                match self.parse_expr() {
                                    Ok(_) => {
                                        *expr =
                                            self.mk_expr_err(expr.span.to(self.prev_token.span));
                                        return Ok(());
                                    }
                                    Err(mut err) => {
                                        *expr = self.mk_expr_err(expr.span);
                                        err.cancel();
                                    }
                                }
                            }
                        }
                        Err(mut err) => {
                            err.cancel();
                        }
                        _ => {}
                    }
                }
            }
        }
        Err(e)
    }

    /// Check to see if a pair of chained operators looks like an attempt at chained comparison,
    /// e.g. `1 < x <= 3`. If so, suggest either splitting the comparison into two, or
    /// parenthesising the leftmost comparison.
    fn attempt_chained_comparison_suggestion(
        &mut self,
        err: &mut DiagnosticBuilder<'_>,
        inner_op: &Expr,
        outer_op: &Spanned<AssocOp>,
    ) -> bool /* advanced the cursor */ {
        if let ExprKind::Binary(op, ref l1, ref r1) = inner_op.kind {
            if let ExprKind::Field(_, ident) = l1.kind {
                if ident.as_str().parse::<i32>().is_err() && !matches!(r1.kind, ExprKind::Lit(_)) {
                    // The parser has encountered `foo.bar<baz`, the likelihood of the turbofish
                    // suggestion being the only one to apply is high.
                    return false;
                }
            }
            let mut enclose = |left: Span, right: Span| {
                err.multipart_suggestion(
                    "parenthesize the comparison",
                    vec![
                        (left.shrink_to_lo(), "(".to_string()),
                        (right.shrink_to_hi(), ")".to_string()),
                    ],
                    Applicability::MaybeIncorrect,
                );
            };
            return match (op.node, &outer_op.node) {
                // `x == y == z`
                (BinOpKind::Eq, AssocOp::Equal) |
                // `x < y < z` and friends.
                (BinOpKind::Lt, AssocOp::Less | AssocOp::LessEqual) |
                (BinOpKind::Le, AssocOp::LessEqual | AssocOp::Less) |
                // `x > y > z` and friends.
                (BinOpKind::Gt, AssocOp::Greater | AssocOp::GreaterEqual) |
                (BinOpKind::Ge, AssocOp::GreaterEqual | AssocOp::Greater) => {
                    let expr_to_str = |e: &Expr| {
                        self.span_to_snippet(e.span)
                            .unwrap_or_else(|_| pprust::expr_to_string(&e))
                    };
                    err.span_suggestion_verbose(
                        inner_op.span.shrink_to_hi(),
                        "split the comparison into two",
                        format!(" && {}", expr_to_str(&r1)),
                        Applicability::MaybeIncorrect,
                    );
                    false // Keep the current parse behavior, where the AST is `(x < y) < z`.
                }
                // `x == y < z`
                (BinOpKind::Eq, AssocOp::Less | AssocOp::LessEqual | AssocOp::Greater | AssocOp::GreaterEqual) => {
                    // Consume `z`/outer-op-rhs.
                    let snapshot = self.clone();
                    match self.parse_expr() {
                        Ok(r2) => {
                            // We are sure that outer-op-rhs could be consumed, the suggestion is
                            // likely correct.
                            enclose(r1.span, r2.span);
                            true
                        }
                        Err(mut expr_err) => {
                            expr_err.cancel();
                            *self = snapshot;
                            false
                        }
                    }
                }
                // `x > y == z`
                (BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge, AssocOp::Equal) => {
                    let snapshot = self.clone();
                    // At this point it is always valid to enclose the lhs in parentheses, no
                    // further checks are necessary.
                    match self.parse_expr() {
                        Ok(_) => {
                            enclose(l1.span, r1.span);
                            true
                        }
                        Err(mut expr_err) => {
                            expr_err.cancel();
                            *self = snapshot;
                            false
                        }
                    }
                }
                _ => false,
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
            |this: &Self, span| Ok(Some(this.mk_expr(span, ExprKind::Err, AttrVec::new())));

        match inner_op.kind {
            ExprKind::Binary(op, ref l1, ref r1) if op.node.is_comparison() => {
                let mut err = self.struct_span_err(
                    vec![op.span, self.prev_token.span],
                    "comparison operators cannot be chained",
                );

                let suggest = |err: &mut DiagnosticBuilder<'_>| {
                    err.span_suggestion_verbose(
                        op.span.shrink_to_lo(),
                        TURBOFISH_SUGGESTION_STR,
                        "::".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                };

                // Include `<` to provide this recommendation even in a case like
                // `Foo<Bar<Baz<Qux, ()>>>`
                if op.node == BinOpKind::Lt && outer_op.node == AssocOp::Less
                    || outer_op.node == AssocOp::Greater
                {
                    if outer_op.node == AssocOp::Less {
                        let snapshot = self.clone();
                        self.bump();
                        // So far we have parsed `foo<bar<`, consume the rest of the type args.
                        let modifiers =
                            [(token::Lt, 1), (token::Gt, -1), (token::BinOp(token::Shr), -2)];
                        self.consume_tts(1, &modifiers);

                        if !&[token::OpenDelim(token::Paren), token::ModSep]
                            .contains(&self.token.kind)
                        {
                            // We don't have `foo< bar >(` or `foo< bar >::`, so we rewind the
                            // parser and bail out.
                            *self = snapshot.clone();
                        }
                    }
                    return if token::ModSep == self.token.kind {
                        // We have some certainty that this was a bad turbofish at this point.
                        // `foo< bar >::`
                        suggest(&mut err);

                        let snapshot = self.clone();
                        self.bump(); // `::`

                        // Consume the rest of the likely `foo<bar>::new()` or return at `foo<bar>`.
                        match self.parse_expr() {
                            Ok(_) => {
                                // 99% certain that the suggestion is correct, continue parsing.
                                err.emit();
                                // FIXME: actually check that the two expressions in the binop are
                                // paths and resynthesize new fn call expression instead of using
                                // `ExprKind::Err` placeholder.
                                mk_err_expr(self, inner_op.span.to(self.prev_token.span))
                            }
                            Err(mut expr_err) => {
                                expr_err.cancel();
                                // Not entirely sure now, but we bubble the error up with the
                                // suggestion.
                                *self = snapshot;
                                Err(err)
                            }
                        }
                    } else if token::OpenDelim(token::Paren) == self.token.kind {
                        // We have high certainty that this was a bad turbofish at this point.
                        // `foo< bar >(`
                        suggest(&mut err);
                        // Consume the fn call arguments.
                        match self.consume_fn_args() {
                            Err(()) => Err(err),
                            Ok(()) => {
                                err.emit();
                                // FIXME: actually check that the two expressions in the binop are
                                // paths and resynthesize new fn call expression instead of using
                                // `ExprKind::Err` placeholder.
                                mk_err_expr(self, inner_op.span.to(self.prev_token.span))
                            }
                        }
                    } else {
                        if !matches!(l1.kind, ExprKind::Lit(_))
                            && !matches!(r1.kind, ExprKind::Lit(_))
                        {
                            // All we know is that this is `foo < bar >` and *nothing* else. Try to
                            // be helpful, but don't attempt to recover.
                            err.help(TURBOFISH_SUGGESTION_STR);
                            err.help("or use `(...)` if you meant to specify fn arguments");
                        }

                        // If it looks like a genuine attempt to chain operators (as opposed to a
                        // misformatted turbofish, for instance), suggest a correct form.
                        if self.attempt_chained_comparison_suggestion(&mut err, inner_op, outer_op)
                        {
                            err.emit();
                            mk_err_expr(self, inner_op.span.to(self.prev_token.span))
                        } else {
                            // These cases cause too many knock-down errors, bail out (#61329).
                            Err(err)
                        }
                    };
                }
                let recover =
                    self.attempt_chained_comparison_suggestion(&mut err, inner_op, outer_op);
                err.emit();
                if recover {
                    return mk_err_expr(self, inner_op.span.to(self.prev_token.span));
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn consume_fn_args(&mut self) -> Result<(), ()> {
        let snapshot = self.clone();
        self.bump(); // `(`

        // Consume the fn call arguments.
        let modifiers =
            [(token::OpenDelim(token::Paren), 1), (token::CloseDelim(token::Paren), -1)];
        self.consume_tts(1, &modifiers);

        if self.token.kind == token::Eof {
            // Not entirely sure that what we consumed were fn arguments, rollback.
            *self = snapshot;
            Err(())
        } else {
            // 99% certain that the suggestion is correct, continue parsing.
            Ok(())
        }
    }

    pub(super) fn maybe_report_ambiguous_plus(
        &mut self,
        allow_plus: AllowPlus,
        impl_dyn_multi: bool,
        ty: &Ty,
    ) {
        if matches!(allow_plus, AllowPlus::No) && impl_dyn_multi {
            let sum_with_parens = format!("({})", pprust::ty_to_string(&ty));
            self.struct_span_err(ty.span, "ambiguous `+` in a type")
                .span_suggestion(
                    ty.span,
                    "use parentheses to disambiguate",
                    sum_with_parens,
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }

    /// Swift lets users write `Ty?` to mean `Option<Ty>`. Parse the construct and recover from it.
    pub(super) fn maybe_recover_from_question_mark(
        &mut self,
        ty: P<Ty>,
        recover_question_mark: RecoverQuestionMark,
    ) -> P<Ty> {
        if let RecoverQuestionMark::No = recover_question_mark {
            return ty;
        }
        if self.token == token::Question {
            self.bump();
            self.struct_span_err(self.prev_token.span, "invalid `?` in type")
                .span_label(self.prev_token.span, "`?` is only allowed on expressions, not types")
                .multipart_suggestion(
                    "if you meant to express that the type might not contain a value, use the `Option` wrapper type",
                    vec![
                        (ty.span.shrink_to_lo(), "Option<".to_string()),
                        (self.prev_token.span, ">".to_string()),
                    ],
                    Applicability::MachineApplicable,
                )
                .emit();
            self.mk_ty(ty.span.to(self.prev_token.span), TyKind::Err)
        } else {
            ty
        }
    }

    pub(super) fn maybe_recover_from_bad_type_plus(
        &mut self,
        allow_plus: AllowPlus,
        ty: &Ty,
    ) -> PResult<'a, ()> {
        // Do not add `+` to expected tokens.
        if matches!(allow_plus, AllowPlus::No) || !self.token.is_like_plus() {
            return Ok(());
        }

        self.bump(); // `+`
        let bounds = self.parse_generic_bounds(None)?;
        let sum_span = ty.span.to(self.prev_token.span);

        let mut err = struct_span_err!(
            self.sess.span_diagnostic,
            sum_span,
            E0178,
            "expected a path on the left-hand side of `+`, not `{}`",
            pprust::ty_to_string(ty)
        );

        match ty.kind {
            TyKind::Rptr(ref lifetime, ref mut_ty) => {
                let sum_with_parens = pprust::to_string(|s| {
                    s.s.word("&");
                    s.print_opt_lifetime(lifetime);
                    s.print_mutability(mut_ty.mutbl, false);
                    s.popen();
                    s.print_type(&mut_ty.ty);
                    s.print_type_bounds(" +", &bounds);
                    s.pclose()
                });
                err.span_suggestion(
                    sum_span,
                    "try adding parentheses",
                    sum_with_parens,
                    Applicability::MachineApplicable,
                );
            }
            TyKind::Ptr(..) | TyKind::BareFn(..) => {
                err.span_label(sum_span, "perhaps you forgot parentheses?");
            }
            _ => {
                err.span_label(sum_span, "expected a path");
            }
        }
        err.emit();
        Ok(())
    }

    /// Tries to recover from associated item paths like `[T]::AssocItem` / `(T, U)::AssocItem`.
    /// Attempts to convert the base expression/pattern/type into a type, parses the `::AssocItem`
    /// tail, and combines them into a `<Ty>::AssocItem` expression/pattern/type.
    pub(super) fn maybe_recover_from_bad_qpath<T: RecoverQPath>(
        &mut self,
        base: P<T>,
        allow_recovery: bool,
    ) -> PResult<'a, P<T>> {
        // Do not add `::` to expected tokens.
        if allow_recovery && self.token == token::ModSep {
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
        self.expect(&token::ModSep)?;

        let mut path = ast::Path { segments: Vec::new(), span: DUMMY_SP, tokens: None };
        self.parse_path_segments(&mut path.segments, T::PATH_STYLE, None)?;
        path.span = ty_span.to(self.prev_token.span);

        let ty_str = self.span_to_snippet(ty_span).unwrap_or_else(|_| pprust::ty_to_string(&ty));
        self.struct_span_err(path.span, "missing angle brackets in associated item path")
            .span_suggestion(
                // This is a best-effort recovery.
                path.span,
                "try",
                format!("<{}>::{}", ty_str, pprust::path_to_string(&path)),
                Applicability::MaybeIncorrect,
            )
            .emit();

        let path_span = ty_span.shrink_to_hi(); // Use an empty path since `position == 0`.
        Ok(P(T::recovered(Some(QSelf { ty, path_span, position: 0 }), path)))
    }

    pub fn maybe_consume_incorrect_semicolon(&mut self, items: &[P<Item>]) -> bool {
        if self.token.kind == TokenKind::Semi {
            self.bump();
            let mut err = self.struct_span_err(self.prev_token.span, "expected item, found `;`");
            err.span_suggestion_short(
                self.prev_token.span,
                "remove this semicolon",
                String::new(),
                Applicability::MachineApplicable,
            );
            if !items.is_empty() {
                let previous_item = &items[items.len() - 1];
                let previous_item_kind_name = match previous_item.kind {
                    // Say "braced struct" because tuple-structs and
                    // braceless-empty-struct declarations do take a semicolon.
                    ItemKind::Struct(..) => Some("braced struct"),
                    ItemKind::Enum(..) => Some("enum"),
                    ItemKind::Trait(..) => Some("trait"),
                    ItemKind::Union(..) => Some("union"),
                    _ => None,
                };
                if let Some(name) = previous_item_kind_name {
                    err.help(&format!("{} declarations are not followed by a semicolon", name));
                }
            }
            err.emit();
            true
        } else {
            false
        }
    }

    /// Creates a `DiagnosticBuilder` for an unexpected token `t` and tries to recover if it is a
    /// closing delimiter.
    pub(super) fn unexpected_try_recover(
        &mut self,
        t: &TokenKind,
    ) -> PResult<'a, bool /* recovered */> {
        let token_str = pprust::token_kind_to_string(t);
        let this_token_str = super::token_descr(&self.token);
        let (prev_sp, sp) = match (&self.token.kind, self.subparser_name) {
            // Point at the end of the macro call when reaching end of macro arguments.
            (token::Eof, Some(_)) => {
                let sp = self.sess.source_map().next_point(self.prev_token.span);
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
                (token::Eof, Some(origin)) => format!("end of {}", origin),
                _ => this_token_str,
            },
        );
        let mut err = self.struct_span_err(sp, &msg);
        let label_exp = format!("expected `{}`", token_str);
        match self.recover_closing_delimiter(&[t.clone()], err) {
            Err(e) => err = e,
            Ok(recovered) => {
                return Ok(recovered);
            }
        }
        let sm = self.sess.source_map();
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
        if self.eat(&token::Semi) {
            return Ok(());
        }
        self.expect(&token::Semi).map(drop) // Error unconditionally
    }

    /// Consumes alternative await syntaxes like `await!(<expr>)`, `await <expr>`,
    /// `await? <expr>`, `await(<expr>)`, and `await { <expr> }`.
    pub(super) fn recover_incorrect_await_syntax(
        &mut self,
        lo: Span,
        await_sp: Span,
        attrs: AttrVec,
    ) -> PResult<'a, P<Expr>> {
        let (hi, expr, is_question) = if self.token == token::Not {
            // Handle `await!(<expr>)`.
            self.recover_await_macro()?
        } else {
            self.recover_await_prefix(await_sp)?
        };
        let sp = self.error_on_incorrect_await(lo, hi, &expr, is_question);
        let kind = match expr.kind {
            // Avoid knock-down errors as we don't know whether to interpret this as `foo().await?`
            // or `foo()?.await` (the very reason we went with postfix syntax 😅).
            ExprKind::Try(_) => ExprKind::Err,
            _ => ExprKind::Await(expr),
        };
        let expr = self.mk_expr(lo.to(sp), kind, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    fn recover_await_macro(&mut self) -> PResult<'a, (Span, P<Expr>, bool)> {
        self.expect(&token::Not)?;
        self.expect(&token::OpenDelim(token::Paren))?;
        let expr = self.parse_expr()?;
        self.expect(&token::CloseDelim(token::Paren))?;
        Ok((self.prev_token.span, expr, false))
    }

    fn recover_await_prefix(&mut self, await_sp: Span) -> PResult<'a, (Span, P<Expr>, bool)> {
        let is_question = self.eat(&token::Question); // Handle `await? <expr>`.
        let expr = if self.token == token::OpenDelim(token::Brace) {
            // Handle `await { <expr> }`.
            // This needs to be handled separately from the next arm to avoid
            // interpreting `await { <expr> }?` as `<expr>?.await`.
            self.parse_block_expr(None, self.token.span, BlockCheckMode::Default, AttrVec::new())
        } else {
            self.parse_expr()
        }
        .map_err(|mut err| {
            err.span_label(await_sp, "while parsing this incorrect await expression");
            err
        })?;
        Ok((expr.span, expr, is_question))
    }

    fn error_on_incorrect_await(&self, lo: Span, hi: Span, expr: &Expr, is_question: bool) -> Span {
        let expr_str =
            self.span_to_snippet(expr.span).unwrap_or_else(|_| pprust::expr_to_string(&expr));
        let suggestion = format!("{}.await{}", expr_str, if is_question { "?" } else { "" });
        let sp = lo.to(hi);
        let app = match expr.kind {
            ExprKind::Try(_) => Applicability::MaybeIncorrect, // `await <expr>?`
            _ => Applicability::MachineApplicable,
        };
        self.struct_span_err(sp, "incorrect use of `await`")
            .span_suggestion(sp, "`await` is a postfix operation", suggestion, app)
            .emit();
        sp
    }

    /// If encountering `future.await()`, consumes and emits an error.
    pub(super) fn recover_from_await_method_call(&mut self) {
        if self.token == token::OpenDelim(token::Paren)
            && self.look_ahead(1, |t| t == &token::CloseDelim(token::Paren))
        {
            // future.await()
            let lo = self.token.span;
            self.bump(); // (
            let sp = lo.to(self.token.span);
            self.bump(); // )
            self.struct_span_err(sp, "incorrect use of `await`")
                .span_suggestion(
                    sp,
                    "`await` is not a method call, remove the parentheses",
                    String::new(),
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }

    pub(super) fn try_macro_suggestion(&mut self) -> PResult<'a, P<Expr>> {
        let is_try = self.token.is_keyword(kw::Try);
        let is_questionmark = self.look_ahead(1, |t| t == &token::Not); //check for !
        let is_open = self.look_ahead(2, |t| t == &token::OpenDelim(token::Paren)); //check for (

        if is_try && is_questionmark && is_open {
            let lo = self.token.span;
            self.bump(); //remove try
            self.bump(); //remove !
            let try_span = lo.to(self.token.span); //we take the try!( span
            self.bump(); //remove (
            let is_empty = self.token == token::CloseDelim(token::Paren); //check if the block is empty
            self.consume_block(token::Paren, ConsumeClosingDelim::No); //eat the block
            let hi = self.token.span;
            self.bump(); //remove )
            let mut err = self.struct_span_err(lo.to(hi), "use of deprecated `try` macro");
            err.note("in the 2018 edition `try` is a reserved keyword, and the `try!()` macro is deprecated");
            let prefix = if is_empty { "" } else { "alternatively, " };
            if !is_empty {
                err.multipart_suggestion(
                    "you can use the `?` operator instead",
                    vec![(try_span, "".to_owned()), (hi, "?".to_owned())],
                    Applicability::MachineApplicable,
                );
            }
            err.span_suggestion(lo.shrink_to_lo(), &format!("{}you can still access the deprecated `try!()` macro using the \"raw identifier\" syntax", prefix), "r#".to_string(), Applicability::MachineApplicable);
            err.emit();
            Ok(self.mk_expr_err(lo.to(hi)))
        } else {
            Err(self.expected_expression_found()) // The user isn't trying to invoke the try! macro
        }
    }

    /// Recovers a situation like `for ( $pat in $expr )`
    /// and suggest writing `for $pat in $expr` instead.
    ///
    /// This should be called before parsing the `$block`.
    pub(super) fn recover_parens_around_for_head(
        &mut self,
        pat: P<Pat>,
        begin_paren: Option<Span>,
    ) -> P<Pat> {
        match (&self.token.kind, begin_paren) {
            (token::CloseDelim(token::Paren), Some(begin_par_sp)) => {
                self.bump();

                self.struct_span_err(
                    MultiSpan::from_spans(vec![begin_par_sp, self.prev_token.span]),
                    "unexpected parentheses surrounding `for` loop head",
                )
                .multipart_suggestion(
                    "remove parentheses in `for` loop",
                    vec![(begin_par_sp, String::new()), (self.prev_token.span, String::new())],
                    // With e.g. `for (x) in y)` this would replace `(x) in y)`
                    // with `x) in y)` which is syntactically invalid.
                    // However, this is prevented before we get here.
                    Applicability::MachineApplicable,
                )
                .emit();

                // Unwrap `(pat)` into `pat` to avoid the `unused_parens` lint.
                pat.and_then(|pat| match pat.kind {
                    PatKind::Paren(pat) => pat,
                    _ => P(pat),
                })
            }
            _ => pat,
        }
    }

    pub(super) fn could_ascription_be_path(&self, node: &ast::ExprKind) -> bool {
        (self.token == token::Lt && // `foo:<bar`, likely a typoed turbofish.
            self.look_ahead(1, |t| t.is_ident() && !t.is_reserved_ident()))
            || self.token.is_ident() &&
            matches!(node, ast::ExprKind::Path(..) | ast::ExprKind::Field(..)) &&
            !self.token.is_reserved_ident() &&           // v `foo:bar(baz)`
            self.look_ahead(1, |t| t == &token::OpenDelim(token::Paren))
            || self.look_ahead(1, |t| t == &token::OpenDelim(token::Brace)) // `foo:bar {`
            || self.look_ahead(1, |t| t == &token::Colon) &&     // `foo:bar::<baz`
            self.look_ahead(2, |t| t == &token::Lt) &&
            self.look_ahead(3, |t| t.is_ident())
            || self.look_ahead(1, |t| t == &token::Colon) &&  // `foo:bar:baz`
            self.look_ahead(2, |t| t.is_ident())
            || self.look_ahead(1, |t| t == &token::ModSep)
                && (self.look_ahead(2, |t| t.is_ident()) ||   // `foo:bar::baz`
            self.look_ahead(2, |t| t == &token::Lt)) // `foo:bar::<baz>`
    }

    pub(super) fn recover_seq_parse_error(
        &mut self,
        delim: token::DelimToken,
        lo: Span,
        result: PResult<'a, P<Expr>>,
    ) -> P<Expr> {
        match result {
            Ok(x) => x,
            Err(mut err) => {
                err.emit();
                // Recover from parse error, callers expect the closing delim to be consumed.
                self.consume_block(delim, ConsumeClosingDelim::Yes);
                self.mk_expr(lo.to(self.prev_token.span), ExprKind::Err, AttrVec::new())
            }
        }
    }

    pub(super) fn recover_closing_delimiter(
        &mut self,
        tokens: &[TokenKind],
        mut err: DiagnosticBuilder<'a>,
    ) -> PResult<'a, bool> {
        let mut pos = None;
        // We want to use the last closing delim that would apply.
        for (i, unmatched) in self.unclosed_delims.iter().enumerate().rev() {
            if tokens.contains(&token::CloseDelim(unmatched.expected_delim))
                && Some(self.token.span) > unmatched.unclosed_span
            {
                pos = Some(i);
            }
        }
        match pos {
            Some(pos) => {
                // Recover and assume that the detected unclosed delimiter was meant for
                // this location. Emit the diagnostic and act as if the delimiter was
                // present for the parser's sake.

                // Don't attempt to recover from this unclosed delimiter more than once.
                let unmatched = self.unclosed_delims.remove(pos);
                let delim = TokenType::Token(token::CloseDelim(unmatched.expected_delim));
                if unmatched.found_delim.is_none() {
                    // We encountered `Eof`, set this fact here to avoid complaining about missing
                    // `fn main()` when we found place to suggest the closing brace.
                    *self.sess.reached_eof.borrow_mut() = true;
                }

                // We want to suggest the inclusion of the closing delimiter where it makes
                // the most sense, which is immediately after the last token:
                //
                //  {foo(bar {}}
                //      ^      ^
                //      |      |
                //      |      help: `)` may belong here
                //      |
                //      unclosed delimiter
                if let Some(sp) = unmatched.unclosed_span {
                    let mut primary_span: Vec<Span> =
                        err.span.primary_spans().iter().cloned().collect();
                    primary_span.push(sp);
                    let mut primary_span: MultiSpan = primary_span.into();
                    for span_label in err.span.span_labels() {
                        if let Some(label) = span_label.label {
                            primary_span.push_span_label(span_label.span, label);
                        }
                    }
                    err.set_span(primary_span);
                    err.span_label(sp, "unclosed delimiter");
                }
                // Backticks should be removed to apply suggestions.
                let mut delim = delim.to_string();
                delim.retain(|c| c != '`');
                err.span_suggestion_short(
                    self.prev_token.span.shrink_to_hi(),
                    &format!("`{}` may belong here", delim),
                    delim,
                    Applicability::MaybeIncorrect,
                );
                if unmatched.found_delim.is_none() {
                    // Encountered `Eof` when lexing blocks. Do not recover here to avoid knockdown
                    // errors which would be emitted elsewhere in the parser and let other error
                    // recovery consume the rest of the file.
                    Err(err)
                } else {
                    err.emit();
                    self.expected_tokens.clear(); // Reduce the number of errors.
                    Ok(true)
                }
            }
            _ => Err(err),
        }
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
                token::OpenDelim(token::DelimToken::Brace) => {
                    brace_depth += 1;
                    self.bump();
                    if break_on_block == BlockMode::Break && brace_depth == 1 && bracket_depth == 0
                    {
                        in_block = true;
                    }
                }
                token::OpenDelim(token::DelimToken::Bracket) => {
                    bracket_depth += 1;
                    self.bump();
                }
                token::CloseDelim(token::DelimToken::Brace) => {
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
                token::CloseDelim(token::DelimToken::Bracket) => {
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
                    debug!("recover_stmt_ return - Semi");
                    break;
                }
                _ => self.bump(),
            }
        }
    }

    pub(super) fn check_for_for_in_in_typo(&mut self, in_span: Span) {
        if self.eat_keyword(kw::In) {
            // a common typo: `for _ in in bar {}`
            self.struct_span_err(self.prev_token.span, "expected iterable, found keyword `in`")
                .span_suggestion_short(
                    in_span.until(self.prev_token.span),
                    "remove the duplicated `in`",
                    String::new(),
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }

    pub(super) fn eat_incorrect_doc_comment_for_param_type(&mut self) {
        if let token::DocComment(..) = self.token.kind {
            self.struct_span_err(
                self.token.span,
                "documentation comments cannot be applied to a function parameter's type",
            )
            .span_label(self.token.span, "doc comments are not allowed here")
            .emit();
            self.bump();
        } else if self.token == token::Pound
            && self.look_ahead(1, |t| *t == token::OpenDelim(token::Bracket))
        {
            let lo = self.token.span;
            // Skip every token until next possible arg.
            while self.token != token::CloseDelim(token::Bracket) {
                self.bump();
            }
            let sp = lo.to(self.token.span);
            self.bump();
            self.struct_span_err(sp, "attributes cannot be applied to a function parameter's type")
                .span_label(sp, "attributes are not allowed here")
                .emit();
        }
    }

    pub(super) fn parameter_without_type(
        &mut self,
        err: &mut DiagnosticBuilder<'_>,
        pat: P<ast::Pat>,
        require_name: bool,
        first_param: bool,
    ) -> Option<Ident> {
        // If we find a pattern followed by an identifier, it could be an (incorrect)
        // C-style parameter declaration.
        if self.check_ident()
            && self.look_ahead(1, |t| *t == token::Comma || *t == token::CloseDelim(token::Paren))
        {
            // `fn foo(String s) {}`
            let ident = self.parse_ident().unwrap();
            let span = pat.span.with_hi(ident.span.hi());

            err.span_suggestion(
                span,
                "declare the type after the parameter binding",
                String::from("<identifier>: <type>"),
                Applicability::HasPlaceholders,
            );
            return Some(ident);
        } else if require_name
            && (self.token == token::Comma
                || self.token == token::Lt
                || self.token == token::CloseDelim(token::Paren))
        {
            let rfc_note = "anonymous parameters are removed in the 2018 edition (see RFC 1685)";

            let (ident, self_sugg, param_sugg, type_sugg, self_span, param_span, type_span) =
                match pat.kind {
                    PatKind::Ident(_, ident, _) => (
                        ident,
                        "self: ".to_string(),
                        ": TypeName".to_string(),
                        "_: ".to_string(),
                        pat.span.shrink_to_lo(),
                        pat.span.shrink_to_hi(),
                        pat.span.shrink_to_lo(),
                    ),
                    // Also catches `fn foo(&a)`.
                    PatKind::Ref(ref inner_pat, mutab)
                        if matches!(inner_pat.clone().into_inner().kind, PatKind::Ident(..)) =>
                    {
                        match inner_pat.clone().into_inner().kind {
                            PatKind::Ident(_, ident, _) => {
                                let mutab = mutab.prefix_str();
                                (
                                    ident,
                                    "self: ".to_string(),
                                    format!("{}: &{}TypeName", ident, mutab),
                                    "_: ".to_string(),
                                    pat.span.shrink_to_lo(),
                                    pat.span,
                                    pat.span.shrink_to_lo(),
                                )
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        // Otherwise, try to get a type and emit a suggestion.
                        if let Some(ty) = pat.to_ty() {
                            err.span_suggestion_verbose(
                                pat.span,
                                "explicitly ignore the parameter name",
                                format!("_: {}", pprust::ty_to_string(&ty)),
                                Applicability::MachineApplicable,
                            );
                            err.note(rfc_note);
                        }

                        return None;
                    }
                };

            // `fn foo(a, b) {}`, `fn foo(a<x>, b<y>) {}` or `fn foo(usize, usize) {}`
            if first_param {
                err.span_suggestion(
                    self_span,
                    "if this is a `self` type, give it a parameter name",
                    self_sugg,
                    Applicability::MaybeIncorrect,
                );
            }
            // Avoid suggesting that `fn foo(HashMap<u32>)` is fixed with a change to
            // `fn foo(HashMap: TypeName<u32>)`.
            if self.token != token::Lt {
                err.span_suggestion(
                    param_span,
                    "if this is a parameter name, give it a type",
                    param_sugg,
                    Applicability::HasPlaceholders,
                );
            }
            err.span_suggestion(
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
        let pat = self.parse_pat_no_top_alt(Some("argument name"))?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;

        struct_span_err!(
            self.diagnostic(),
            pat.span,
            E0642,
            "patterns aren't allowed in methods without bodies",
        )
        .span_suggestion_short(
            pat.span,
            "give this argument a name or use an underscore to ignore it",
            "_".to_owned(),
            Applicability::MachineApplicable,
        )
        .emit();

        // Pretend the pattern is `_`, to avoid duplicate errors from AST validation.
        let pat =
            P(Pat { kind: PatKind::Wild, span: pat.span, id: ast::DUMMY_NODE_ID, tokens: None });
        Ok((pat, ty))
    }

    pub(super) fn recover_bad_self_param(&mut self, mut param: Param) -> PResult<'a, Param> {
        let sp = param.pat.span;
        param.ty.kind = TyKind::Err;
        self.struct_span_err(sp, "unexpected `self` parameter in function")
            .span_label(sp, "must be the first parameter of an associated function")
            .emit();
        Ok(param)
    }

    pub(super) fn consume_block(
        &mut self,
        delim: token::DelimToken,
        consume_close: ConsumeClosingDelim,
    ) {
        let mut brace_depth = 0;
        loop {
            if self.eat(&token::OpenDelim(delim)) {
                brace_depth += 1;
            } else if self.check(&token::CloseDelim(delim)) {
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
            } else if self.token == token::Eof || self.eat(&token::CloseDelim(token::NoDelim)) {
                return;
            } else {
                self.bump();
            }
        }
    }

    pub(super) fn expected_expression_found(&self) -> DiagnosticBuilder<'a> {
        let (span, msg) = match (&self.token.kind, self.subparser_name) {
            (&token::Eof, Some(origin)) => {
                let sp = self.sess.source_map().next_point(self.prev_token.span);
                (sp, format!("expected expression, found end of {}", origin))
            }
            _ => (
                self.token.span,
                format!("expected expression, found {}", super::token_descr(&self.token),),
            ),
        };
        let mut err = self.struct_span_err(span, &msg);
        let sp = self.sess.source_map().start_point(self.token.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            self.sess.expr_parentheses_needed(&mut err, *sp);
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
            if let Some((_, val)) = modifier.iter().find(|(t, _)| *t == self.token.kind) {
                acc += *val;
            }
            if self.token.kind == token::Eof {
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
    pub(super) fn deduplicate_recovered_params_names(&self, fn_inputs: &mut Vec<Param>) {
        let mut seen_inputs = FxHashSet::default();
        for input in fn_inputs.iter_mut() {
            let opt_ident = if let (PatKind::Ident(_, ident, _), TyKind::Err) =
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
    pub fn handle_ambiguous_unbraced_const_arg(
        &mut self,
        args: &mut Vec<AngleBracketedArg>,
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
        let mut err = self.struct_span_err(
            self.token.span,
            &format!("expected one of `,` or `>`, found {}", super::token_descr(&self.token)),
        );
        err.span_label(self.token.span, "expected one of `,` or `>`");
        match self.recover_const_arg(arg.span(), err) {
            Ok(arg) => {
                args.push(AngleBracketedArg::Arg(arg));
                if self.eat(&token::Comma) {
                    return Ok(true); // Continue
                }
            }
            Err(mut err) => {
                args.push(arg);
                // We will emit a more generic error later.
                err.delay_as_bug();
            }
        }
        return Ok(false); // Don't continue.
    }

    /// Attempt to parse a generic const argument that has not been enclosed in braces.
    /// There are a limited number of expressions that are permitted without being encoded
    /// in braces:
    /// - Literals.
    /// - Single-segment paths (i.e. standalone generic const parameters).
    /// All other expressions that can be parsed will emit an error suggesting the expression be
    /// wrapped in braces.
    pub fn handle_unambiguous_unbraced_const_arg(&mut self) -> PResult<'a, P<Expr>> {
        let start = self.token.span;
        let expr = self.parse_expr_res(Restrictions::CONST_EXPR, None).map_err(|mut err| {
            err.span_label(
                start.shrink_to_lo(),
                "while parsing a const generic argument starting here",
            );
            err
        })?;
        if !self.expr_is_valid_const_arg(&expr) {
            self.struct_span_err(
                expr.span,
                "expressions must be enclosed in braces to be used as const generic \
                    arguments",
            )
            .multipart_suggestion(
                "enclose the `const` expression in braces",
                vec![
                    (expr.span.shrink_to_lo(), "{ ".to_string()),
                    (expr.span.shrink_to_hi(), " }".to_string()),
                ],
                Applicability::MachineApplicable,
            )
            .emit();
        }
        Ok(expr)
    }

    fn recover_const_param_decl(
        &mut self,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Option<GenericArg>> {
        let snapshot = self.clone();
        let param = match self.parse_const_param(vec![]) {
            Ok(param) => param,
            Err(mut err) => {
                err.cancel();
                *self = snapshot;
                return Err(err);
            }
        };
        let mut err =
            self.struct_span_err(param.span(), "unexpected `const` parameter declaration");
        err.span_label(param.span(), "expected a `const` expression, not a parameter declaration");
        if let (Some(generics), Ok(snippet)) =
            (ty_generics, self.sess.source_map().span_to_snippet(param.span()))
        {
            let (span, sugg) = match &generics.params[..] {
                [] => (generics.span, format!("<{}>", snippet)),
                [.., generic] => (generic.span().shrink_to_hi(), format!(", {}", snippet)),
            };
            err.multipart_suggestion(
                "`const` parameters must be declared for the `impl`",
                vec![(span, sugg), (param.span(), param.ident.to_string())],
                Applicability::MachineApplicable,
            );
        }
        let value = self.mk_expr_err(param.span());
        err.emit();
        return Ok(Some(GenericArg::Const(AnonConst { id: ast::DUMMY_NODE_ID, value })));
    }

    pub fn recover_const_param_declaration(
        &mut self,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Option<GenericArg>> {
        // We have to check for a few different cases.
        if let Ok(arg) = self.recover_const_param_decl(ty_generics) {
            return Ok(arg);
        }

        // We haven't consumed `const` yet.
        let start = self.token.span;
        self.bump(); // `const`

        // Detect and recover from the old, pre-RFC2000 syntax for const generics.
        let mut err = self
            .struct_span_err(start, "expected lifetime, type, or constant, found keyword `const`");
        if self.check_const_arg() {
            err.span_suggestion_verbose(
                start.until(self.token.span),
                "the `const` keyword is only needed in the definition of the type",
                String::new(),
                Applicability::MaybeIncorrect,
            );
            err.emit();
            Ok(Some(GenericArg::Const(self.parse_const_arg()?)))
        } else {
            let after_kw_const = self.token.span;
            self.recover_const_arg(after_kw_const, err).map(Some)
        }
    }

    /// Try to recover from possible generic const argument without `{` and `}`.
    ///
    /// When encountering code like `foo::< bar + 3 >` or `foo::< bar - baz >` we suggest
    /// `foo::<{ bar + 3 }>` and `foo::<{ bar - baz }>`, respectively. We only provide a suggestion
    /// if we think that that the resulting expression would be well formed.
    pub fn recover_const_arg(
        &mut self,
        start: Span,
        mut err: DiagnosticBuilder<'a>,
    ) -> PResult<'a, GenericArg> {
        let is_op = AssocOp::from_token(&self.token)
            .and_then(|op| {
                if let AssocOp::Greater
                | AssocOp::Less
                | AssocOp::ShiftRight
                | AssocOp::GreaterEqual
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
            .is_some();
        // This will be true when a trait object type `Foo +` or a path which was a `const fn` with
        // type params has been parsed.
        let was_op =
            matches!(self.prev_token.kind, token::BinOp(token::Plus | token::Shr) | token::Gt);
        if !is_op && !was_op {
            // We perform these checks and early return to avoid taking a snapshot unnecessarily.
            return Err(err);
        }
        let snapshot = self.clone();
        if is_op {
            self.bump();
        }
        match self.parse_expr_res(Restrictions::CONST_EXPR, None) {
            Ok(expr) => {
                // Find a mistake like `MyTrait<Assoc == S::Assoc>`.
                if token::EqEq == snapshot.token.kind {
                    err.span_suggestion(
                        snapshot.token.span,
                        "if you meant to use an associated type binding, replace `==` with `=`",
                        "=".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                    let value = self.mk_expr_err(start.to(expr.span));
                    err.emit();
                    return Ok(GenericArg::Const(AnonConst { id: ast::DUMMY_NODE_ID, value }));
                } else if token::Comma == self.token.kind || self.token.kind.should_end_const_arg()
                {
                    // Avoid the following output by checking that we consumed a full const arg:
                    // help: expressions must be enclosed in braces to be used as const generic
                    //       arguments
                    //    |
                    // LL |     let sr: Vec<{ (u32, _, _) = vec![] };
                    //    |                 ^                      ^
                    err.multipart_suggestion(
                        "expressions must be enclosed in braces to be used as const generic \
                         arguments",
                        vec![
                            (start.shrink_to_lo(), "{ ".to_string()),
                            (expr.span.shrink_to_hi(), " }".to_string()),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                    let value = self.mk_expr_err(start.to(expr.span));
                    err.emit();
                    return Ok(GenericArg::Const(AnonConst { id: ast::DUMMY_NODE_ID, value }));
                }
            }
            Err(mut err) => {
                err.cancel();
            }
        }
        *self = snapshot;
        Err(err)
    }

    /// Get the diagnostics for the cases where `move async` is found.
    ///
    /// `move_async_span` starts at the 'm' of the move keyword and ends with the 'c' of the async keyword
    pub(super) fn incorrect_move_async_order_found(
        &self,
        move_async_span: Span,
    ) -> DiagnosticBuilder<'a> {
        let mut err =
            self.struct_span_err(move_async_span, "the order of `move` and `async` is incorrect");
        err.span_suggestion_verbose(
            move_async_span,
            "try switching the order",
            "async move".to_owned(),
            Applicability::MaybeIncorrect,
        );
        err
    }

    /// Some special error handling for the "top-level" patterns in a match arm,
    /// `for` loop, `let`, &c. (in contrast to subpatterns within such).
    crate fn maybe_recover_colon_colon_in_pat_typo(
        &mut self,
        mut first_pat: P<Pat>,
        ra: RecoverColon,
        expected: Expected,
    ) -> P<Pat> {
        if RecoverColon::Yes != ra || token::Colon != self.token.kind {
            return first_pat;
        }
        if !matches!(first_pat.kind, PatKind::Ident(_, _, None) | PatKind::Path(..))
            || !self.look_ahead(1, |token| token.is_ident() && !token.is_reserved_ident())
        {
            return first_pat;
        }
        // The pattern looks like it might be a path with a `::` -> `:` typo:
        // `match foo { bar:baz => {} }`
        let span = self.token.span;
        // We only emit "unexpected `:`" error here if we can successfully parse the
        // whole pattern correctly in that case.
        let snapshot = self.clone();

        // Create error for "unexpected `:`".
        match self.expected_one_of_not_found(&[], &[]) {
            Err(mut err) => {
                self.bump(); // Skip the `:`.
                match self.parse_pat_no_top_alt(expected) {
                    Err(mut inner_err) => {
                        // Carry on as if we had not done anything, callers will emit a
                        // reasonable error.
                        inner_err.cancel();
                        err.cancel();
                        *self = snapshot;
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
                            PatKind::Ident(BindingMode::ByValue(Mutability::Not), ident, None) => {
                                match &first_pat.kind {
                                    PatKind::Ident(_, old_ident, _) => {
                                        let path = PatKind::Path(
                                            None,
                                            Path {
                                                span: new_span,
                                                segments: vec![
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
                            err.span_suggestion(
                                span,
                                "maybe write a path separator here",
                                "::".to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            first_pat = self.mk_pat(new_span, PatKind::Wild);
                        }
                        err.emit();
                    }
                }
            }
            _ => {
                // Carry on as if we had not done anything. This should be unreachable.
                *self = snapshot;
            }
        };
        first_pat
    }

    /// Some special error handling for the "top-level" patterns in a match arm,
    /// `for` loop, `let`, &c. (in contrast to subpatterns within such).
    crate fn maybe_recover_unexpected_comma(
        &mut self,
        lo: Span,
        rc: RecoverComma,
    ) -> PResult<'a, ()> {
        if rc == RecoverComma::No || self.token != token::Comma {
            return Ok(());
        }

        // An unexpected comma after a top-level pattern is a clue that the
        // user (perhaps more accustomed to some other language) forgot the
        // parentheses in what should have been a tuple pattern; return a
        // suggestion-enhanced error here rather than choking on the comma later.
        let comma_span = self.token.span;
        self.bump();
        if let Err(mut err) = self.skip_pat_list() {
            // We didn't expect this to work anyway; we just wanted to advance to the
            // end of the comma-sequence so we know the span to suggest parenthesizing.
            err.cancel();
        }
        let seq_span = lo.to(self.prev_token.span);
        let mut err = self.struct_span_err(comma_span, "unexpected `,` in pattern");
        if let Ok(seq_snippet) = self.span_to_snippet(seq_span) {
            const MSG: &str = "try adding parentheses to match on a tuple...";

            err.span_suggestion(
                seq_span,
                MSG,
                format!("({})", seq_snippet),
                Applicability::MachineApplicable,
            );
            err.span_suggestion(
                seq_span,
                "...or a vertical bar to match on multiple alternatives",
                seq_snippet.replace(',', " |"),
                Applicability::MachineApplicable,
            );
        }
        Err(err)
    }

    /// Parse and throw away a parenthesized comma separated
    /// sequence of patterns until `)` is reached.
    fn skip_pat_list(&mut self) -> PResult<'a, ()> {
        while !self.check(&token::CloseDelim(token::Paren)) {
            self.parse_pat_no_top_alt(None)?;
            if !self.eat(&token::Comma) {
                return Ok(());
            }
        }
        Ok(())
    }
}
