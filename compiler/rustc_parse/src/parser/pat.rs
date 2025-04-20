use std::ops::Bound;

use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::token::NtPatKind::*;
use rustc_ast::token::{self, IdentIsRaw, MetaVarKind, Token};
use rustc_ast::util::parser::ExprPrecedence;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{
    self as ast, Arm, AttrVec, BindingMode, ByRef, Expr, ExprKind, LocalKind, MacCall, Mutability,
    Pat, PatField, PatFieldsRest, PatKind, Path, QSelf, RangeEnd, RangeSyntax, Stmt, StmtKind,
};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, Diag, DiagArgValue, PResult, StashKey};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::source_map::{Spanned, respan};
use rustc_span::{BytePos, ErrorGuaranteed, Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use super::{ForceCollect, Parser, PathStyle, Restrictions, Trailing, UsePreAttrPos};
use crate::errors::{
    self, AmbiguousRangePattern, AtDotDotInStructPattern, AtInStructPattern,
    DotDotDotForRemainingFields, DotDotDotRangeToPatternNotAllowed, DotDotDotRestPattern,
    EnumPatternInsteadOfIdentifier, ExpectedBindingLeftOfAt, ExpectedCommaAfterPatternField,
    GenericArgsInPatRequireTurbofishSyntax, InclusiveRangeExtraEquals, InclusiveRangeMatchArrow,
    InclusiveRangeNoEnd, InvalidMutInPattern, ParenRangeSuggestion, PatternOnWrongSideOfAt,
    RemoveLet, RepeatedMutInPattern, SwitchRefBoxOrder, TopLevelOrPatternNotAllowed,
    TopLevelOrPatternNotAllowedSugg, TrailingVertNotAllowed, UnexpectedExpressionInPattern,
    UnexpectedExpressionInPatternSugg, UnexpectedLifetimeInPattern, UnexpectedParenInRangePat,
    UnexpectedParenInRangePatSugg, UnexpectedVertVertBeforeFunctionParam,
    UnexpectedVertVertInPattern, WrapInParens,
};
use crate::parser::expr::{DestructuredFloat, could_be_unclosed_char_literal};
use crate::{exp, maybe_recover_from_interpolated_ty_qpath};

#[derive(PartialEq, Copy, Clone)]
pub enum Expected {
    ParameterName,
    ArgumentName,
    Identifier,
    BindingPattern,
}

impl Expected {
    // FIXME(#100717): migrate users of this to proper localization
    fn to_string_or_fallback(expected: Option<Expected>) -> &'static str {
        match expected {
            Some(Expected::ParameterName) => "parameter name",
            Some(Expected::ArgumentName) => "argument name",
            Some(Expected::Identifier) => "identifier",
            Some(Expected::BindingPattern) => "binding pattern",
            None => "pattern",
        }
    }
}

const WHILE_PARSING_OR_MSG: &str = "while parsing this or-pattern starting here";

/// Whether or not to recover a `,` when parsing or-patterns.
#[derive(PartialEq, Copy, Clone)]
pub enum RecoverComma {
    Yes,
    No,
}

/// Whether or not to recover a `:` when parsing patterns that were meant to be paths.
#[derive(PartialEq, Copy, Clone)]
pub enum RecoverColon {
    Yes,
    No,
}

/// Whether or not to recover a `a, b` when parsing patterns as `(a, b)` or that *and* `a | b`.
#[derive(PartialEq, Copy, Clone)]
pub enum CommaRecoveryMode {
    LikelyTuple,
    EitherTupleOrPipe,
}

/// The result of `eat_or_separator`. We want to distinguish which case we are in to avoid
/// emitting duplicate diagnostics.
#[derive(Debug, Clone, Copy)]
enum EatOrResult {
    /// We recovered from a trailing vert.
    TrailingVert,
    /// We ate an `|` (or `||` and recovered).
    AteOr,
    /// We did not eat anything (i.e. the current token is not `|` or `||`).
    None,
}

/// The syntax location of a given pattern. Used for diagnostics.
#[derive(Clone, Copy)]
pub enum PatternLocation {
    LetBinding,
    FunctionParameter,
}

impl<'a> Parser<'a> {
    /// Parses a pattern.
    ///
    /// Corresponds to `Pattern` in RFC 3637 and admits guard patterns at the top level.
    /// Used when parsing patterns in all cases where neither `PatternNoTopGuard` nor
    /// `PatternNoTopAlt` (see below) are used.
    pub fn parse_pat_allow_top_guard(
        &mut self,
        expected: Option<Expected>,
        rc: RecoverComma,
        ra: RecoverColon,
        rt: CommaRecoveryMode,
    ) -> PResult<'a, P<Pat>> {
        let pat = self.parse_pat_no_top_guard(expected, rc, ra, rt)?;

        if self.eat_keyword(exp!(If)) {
            let cond = self.parse_expr()?;
            // Feature-gate guard patterns
            self.psess.gated_spans.gate(sym::guard_patterns, cond.span);
            let span = pat.span.to(cond.span);
            Ok(self.mk_pat(span, PatKind::Guard(pat, cond)))
        } else {
            Ok(pat)
        }
    }

    /// Parses a pattern.
    ///
    /// Corresponds to `PatternNoTopAlt` in RFC 3637 and does not admit or-patterns
    /// or guard patterns at the top level. Used when parsing the parameters of lambda
    /// expressions, functions, function pointers, and `pat_param` macro fragments.
    pub fn parse_pat_no_top_alt(
        &mut self,
        expected: Option<Expected>,
        syntax_loc: Option<PatternLocation>,
    ) -> PResult<'a, P<Pat>> {
        self.parse_pat_with_range_pat(true, expected, syntax_loc)
    }

    /// Parses a pattern.
    ///
    /// Corresponds to `PatternNoTopGuard` in RFC 3637 and allows or-patterns, but not
    /// guard patterns, at the top level. Used for parsing patterns in `pat` fragments (until
    /// the next edition) and `let`, `if let`, and `while let` expressions.
    ///
    /// Note that after the FCP in <https://github.com/rust-lang/rust/issues/81415>,
    /// a leading vert is allowed in nested or-patterns, too. This allows us to
    /// simplify the grammar somewhat.
    pub fn parse_pat_no_top_guard(
        &mut self,
        expected: Option<Expected>,
        rc: RecoverComma,
        ra: RecoverColon,
        rt: CommaRecoveryMode,
    ) -> PResult<'a, P<Pat>> {
        self.parse_pat_no_top_guard_inner(expected, rc, ra, rt, None).map(|(pat, _)| pat)
    }

    /// Returns the pattern and a bool indicating whether we recovered from a trailing vert (true =
    /// recovered).
    fn parse_pat_no_top_guard_inner(
        &mut self,
        expected: Option<Expected>,
        rc: RecoverComma,
        ra: RecoverColon,
        rt: CommaRecoveryMode,
        syntax_loc: Option<PatternLocation>,
    ) -> PResult<'a, (P<Pat>, bool)> {
        // Keep track of whether we recovered from a trailing vert so that we can avoid duplicated
        // suggestions (which bothers rustfix).
        //
        // Allow a '|' before the pats (RFCs 1925, 2530, and 2535).
        let (leading_vert_span, mut trailing_vert) = match self.eat_or_separator(None) {
            EatOrResult::AteOr => (Some(self.prev_token.span), false),
            EatOrResult::TrailingVert => (None, true),
            EatOrResult::None => (None, false),
        };

        // Parse the first pattern (`p_0`).
        let mut first_pat = match self.parse_pat_no_top_alt(expected, syntax_loc) {
            Ok(pat) => pat,
            Err(err)
                if self.token.is_reserved_ident()
                    && !self.token.is_keyword(kw::In)
                    && !self.token.is_keyword(kw::If) =>
            {
                err.emit();
                self.bump();
                self.mk_pat(self.token.span, PatKind::Wild)
            }
            Err(err) => return Err(err),
        };
        if rc == RecoverComma::Yes && !first_pat.could_be_never_pattern() {
            self.maybe_recover_unexpected_comma(first_pat.span, rt)?;
        }

        // If the next token is not a `|`,
        // this is not an or-pattern and we should exit here.
        if !self.check(exp!(Or)) && self.token != token::OrOr {
            // If we parsed a leading `|` which should be gated,
            // then we should really gate the leading `|`.
            // This complicated procedure is done purely for diagnostics UX.

            // Check if the user wrote `foo:bar` instead of `foo::bar`.
            if ra == RecoverColon::Yes {
                first_pat = self.maybe_recover_colon_colon_in_pat_typo(first_pat, expected);
            }

            if let Some(leading_vert_span) = leading_vert_span {
                // If there was a leading vert, treat this as an or-pattern. This improves
                // diagnostics.
                let span = leading_vert_span.to(self.prev_token.span);
                return Ok((self.mk_pat(span, PatKind::Or(thin_vec![first_pat])), trailing_vert));
            }

            return Ok((first_pat, trailing_vert));
        }

        // Parse the patterns `p_1 | ... | p_n` where `n > 0`.
        let lo = leading_vert_span.unwrap_or(first_pat.span);
        let mut pats = thin_vec![first_pat];
        loop {
            match self.eat_or_separator(Some(lo)) {
                EatOrResult::AteOr => {}
                EatOrResult::None => break,
                EatOrResult::TrailingVert => {
                    trailing_vert = true;
                    break;
                }
            }
            let pat = self.parse_pat_no_top_alt(expected, syntax_loc).map_err(|mut err| {
                err.span_label(lo, WHILE_PARSING_OR_MSG);
                err
            })?;
            if rc == RecoverComma::Yes && !pat.could_be_never_pattern() {
                self.maybe_recover_unexpected_comma(pat.span, rt)?;
            }
            pats.push(pat);
        }
        let or_pattern_span = lo.to(self.prev_token.span);

        Ok((self.mk_pat(or_pattern_span, PatKind::Or(pats)), trailing_vert))
    }

    /// Parse a pattern and (maybe) a `Colon` in positions where a pattern may be followed by a
    /// type annotation (e.g. for `let` bindings or `fn` params).
    ///
    /// Generally, this corresponds to `pat_no_top_alt` followed by an optional `Colon`. It will
    /// eat the `Colon` token if one is present.
    ///
    /// The return value represents the parsed pattern and `true` if a `Colon` was parsed (`false`
    /// otherwise).
    pub(super) fn parse_pat_before_ty(
        &mut self,
        expected: Option<Expected>,
        rc: RecoverComma,
        syntax_loc: PatternLocation,
    ) -> PResult<'a, (P<Pat>, bool)> {
        // We use `parse_pat_allow_top_alt` regardless of whether we actually want top-level
        // or-patterns so that we can detect when a user tries to use it. This allows us to print a
        // better error message.
        let (pat, trailing_vert) = self.parse_pat_no_top_guard_inner(
            expected,
            rc,
            RecoverColon::No,
            CommaRecoveryMode::LikelyTuple,
            Some(syntax_loc),
        )?;
        let colon = self.eat(exp!(Colon));

        if let PatKind::Or(pats) = &pat.kind {
            let span = pat.span;
            let sub = if pats.len() == 1 {
                Some(TopLevelOrPatternNotAllowedSugg::RemoveLeadingVert {
                    span: span.with_hi(span.lo() + BytePos(1)),
                })
            } else {
                Some(TopLevelOrPatternNotAllowedSugg::WrapInParens {
                    span,
                    suggestion: WrapInParens { lo: span.shrink_to_lo(), hi: span.shrink_to_hi() },
                })
            };

            let err = self.dcx().create_err(match syntax_loc {
                PatternLocation::LetBinding => {
                    TopLevelOrPatternNotAllowed::LetBinding { span, sub }
                }
                PatternLocation::FunctionParameter => {
                    TopLevelOrPatternNotAllowed::FunctionParameter { span, sub }
                }
            });
            if trailing_vert {
                err.delay_as_bug();
            } else {
                err.emit();
            }
        }

        Ok((pat, colon))
    }

    /// Parse the pattern for a function or function pointer parameter, followed by a colon.
    ///
    /// The return value represents the parsed pattern and `true` if a `Colon` was parsed (`false`
    /// otherwise).
    pub(super) fn parse_fn_param_pat_colon(&mut self) -> PResult<'a, (P<Pat>, bool)> {
        // In order to get good UX, we first recover in the case of a leading vert for an illegal
        // top-level or-pat. Normally, this means recovering both `|` and `||`, but in this case,
        // a leading `||` probably doesn't indicate an or-pattern attempt, so we handle that
        // separately.
        if let token::OrOr = self.token.kind {
            self.dcx().emit_err(UnexpectedVertVertBeforeFunctionParam { span: self.token.span });
            self.bump();
        }

        self.parse_pat_before_ty(
            Some(Expected::ParameterName),
            RecoverComma::No,
            PatternLocation::FunctionParameter,
        )
    }

    /// Eat the or-pattern `|` separator.
    /// If instead a `||` token is encountered, recover and pretend we parsed `|`.
    fn eat_or_separator(&mut self, lo: Option<Span>) -> EatOrResult {
        if self.recover_trailing_vert(lo) {
            EatOrResult::TrailingVert
        } else if self.token.kind == token::OrOr {
            // Found `||`; Recover and pretend we parsed `|`.
            self.dcx().emit_err(UnexpectedVertVertInPattern { span: self.token.span, start: lo });
            self.bump();
            EatOrResult::AteOr
        } else if self.eat(exp!(Or)) {
            EatOrResult::AteOr
        } else {
            EatOrResult::None
        }
    }

    /// Recover if `|` or `||` is the current token and we have one of the
    /// tokens `=>`, `if`, `=`, `:`, `;`, `,`, `]`, `)`, or `}` ahead of us.
    ///
    /// These tokens all indicate that we reached the end of the or-pattern
    /// list and can now reliably say that the `|` was an illegal trailing vert.
    /// Note that there are more tokens such as `@` for which we know that the `|`
    /// is an illegal parse. However, the user's intent is less clear in that case.
    fn recover_trailing_vert(&mut self, lo: Option<Span>) -> bool {
        let is_end_ahead = self.look_ahead(1, |token| {
            matches!(
                &token.uninterpolate().kind,
                token::FatArrow // e.g. `a | => 0,`.
                | token::Ident(kw::If, token::IdentIsRaw::No) // e.g. `a | if expr`.
                | token::Eq // e.g. `let a | = 0`.
                | token::Semi // e.g. `let a |;`.
                | token::Colon // e.g. `let a | :`.
                | token::Comma // e.g. `let (a |,)`.
                | token::CloseBracket // e.g. `let [a | ]`.
                | token::CloseParen // e.g. `let (a | )`.
                | token::CloseBrace // e.g. `let A { f: a | }`.
            )
        });
        match (is_end_ahead, &self.token.kind) {
            (true, token::Or | token::OrOr) => {
                // A `|` or possibly `||` token shouldn't be here. Ban it.
                self.dcx().emit_err(TrailingVertNotAllowed {
                    span: self.token.span,
                    start: lo,
                    token: self.token,
                    note_double_vert: self.token.kind == token::OrOr,
                });
                self.bump();
                true
            }
            _ => false,
        }
    }

    /// Ensures that the last parsed pattern (or pattern range bound) is not followed by an expression.
    ///
    /// `is_end_bound` indicates whether the last parsed thing was the end bound of a range pattern (see [`parse_pat_range_end`](Self::parse_pat_range_end))
    /// in order to say "expected a pattern range bound" instead of "expected a pattern";
    /// ```text
    /// 0..=1 + 2
    ///     ^^^^^
    /// ```
    /// Only the end bound is spanned in this case, and this function has no idea if there was a `..=` before `pat_span`, hence the parameter.
    ///
    /// This function returns `Some` if a trailing expression was recovered, and said expression's span.
    #[must_use = "the pattern must be discarded as `PatKind::Err` if this function returns Some"]
    fn maybe_recover_trailing_expr(
        &mut self,
        pat_span: Span,
        is_end_bound: bool,
    ) -> Option<(ErrorGuaranteed, Span)> {
        if self.prev_token.is_keyword(kw::Underscore) || !self.may_recover() {
            // Don't recover anything after an `_` or if recovery is disabled.
            return None;
        }

        // Returns `true` iff `token` is an unsuffixed integer.
        let is_one_tuple_index = |_: &Self, token: &Token| -> bool {
            use token::{Lit, LitKind};

            matches!(
                token.kind,
                token::Literal(Lit { kind: LitKind::Integer, symbol: _, suffix: None })
            )
        };

        // Returns `true` iff `token` is an unsuffixed `x.y` float.
        let is_two_tuple_indexes = |this: &Self, token: &Token| -> bool {
            use token::{Lit, LitKind};

            if let token::Literal(Lit { kind: LitKind::Float, symbol, suffix: None }) = token.kind
                && let DestructuredFloat::MiddleDot(..) = this.break_up_float(symbol, token.span)
            {
                true
            } else {
                false
            }
        };

        // Check for `.hello` or `.0`.
        let has_dot_expr = self.check_noexpect(&token::Dot) // `.`
            && self.look_ahead(1, |tok| {
                tok.is_ident() // `hello`
                || is_one_tuple_index(&self, &tok) // `0`
                || is_two_tuple_indexes(&self, &tok) // `0.0`
            });

        // Check for operators.
        // `|` is excluded as it is used in pattern alternatives and lambdas,
        // `?` is included for error propagation,
        // `[` is included for indexing operations,
        // `[]` is excluded as `a[]` isn't an expression and should be recovered as `a, []` (cf. `tests/ui/parser/pat-lt-bracket-7.rs`),
        // `as` is included for type casts
        let has_trailing_operator = matches!(
                self.token.kind,
                token::Plus | token::Minus | token::Star | token::Slash | token::Percent
                | token::Caret | token::And | token::Shl | token::Shr // excludes `Or`
            )
            || self.token == token::Question
            || (self.token == token::OpenBracket
                && self.look_ahead(1, |t| *t != token::CloseBracket)) // excludes `[]`
            || self.token.is_keyword(kw::As);

        if !has_dot_expr && !has_trailing_operator {
            // Nothing to recover here.
            return None;
        }

        // Let's try to parse an expression to emit a better diagnostic.
        let mut snapshot = self.create_snapshot_for_diagnostic();
        snapshot.restrictions.insert(Restrictions::IS_PAT);

        // Parse `?`, `.f`, `(arg0, arg1, ...)` or `[expr]` until they've all been eaten.
        let Ok(expr) = snapshot
            .parse_expr_dot_or_call_with(
                AttrVec::new(),
                self.mk_expr(pat_span, ExprKind::Dummy), // equivalent to transforming the parsed pattern into an `Expr`
                pat_span,
            )
            .map_err(|err| err.cancel())
        else {
            // We got a trailing method/operator, but that wasn't an expression.
            return None;
        };

        // Parse an associative expression such as `+ expr`, `% expr`, ...
        // Assignments, ranges and `|` are disabled by [`Restrictions::IS_PAT`].
        let Ok((expr, _)) = snapshot
            .parse_expr_assoc_rest_with(Bound::Unbounded, false, expr)
            .map_err(|err| err.cancel())
        else {
            // We got a trailing method/operator, but that wasn't an expression.
            return None;
        };

        // We got a valid expression.
        self.restore_snapshot(snapshot);
        self.restrictions.remove(Restrictions::IS_PAT);

        let is_bound = is_end_bound
            // is_start_bound: either `..` or `)..`
            || self.token.is_range_separator()
            || self.token == token::CloseParen
                && self.look_ahead(1, Token::is_range_separator);

        let span = expr.span;

        Some((
            self.dcx()
                .create_err(UnexpectedExpressionInPattern {
                    span,
                    is_bound,
                    expr_precedence: expr.precedence(),
                })
                .stash(span, StashKey::ExprInPat)
                .unwrap(),
            span,
        ))
    }

    /// Called by [`Parser::parse_stmt_without_recovery`], used to add statement-aware subdiagnostics to the errors stashed
    /// by [`Parser::maybe_recover_trailing_expr`].
    pub(super) fn maybe_augment_stashed_expr_in_pats_with_suggestions(&mut self, stmt: &Stmt) {
        if self.dcx().has_errors().is_none() {
            // No need to walk the statement if there's no stashed errors.
            return;
        }

        struct PatVisitor<'a> {
            /// `self`
            parser: &'a Parser<'a>,
            /// The freshly-parsed statement.
            stmt: &'a Stmt,
            /// The current match arm (for arm guard suggestions).
            arm: Option<&'a Arm>,
            /// The current struct field (for variable name suggestions).
            field: Option<&'a PatField>,
        }

        impl<'a> PatVisitor<'a> {
            /// Looks for stashed [`StashKey::ExprInPat`] errors in `stash_span`, and emit them with suggestions.
            /// `stash_span` is contained in `expr_span`, the latter being larger in borrow patterns;
            /// ```txt
            /// &mut x.y
            /// -----^^^ `stash_span`
            /// |
            /// `expr_span`
            /// ```
            /// `is_range_bound` is used to exclude arm guard suggestions in range pattern bounds.
            fn maybe_add_suggestions_then_emit(
                &self,
                stash_span: Span,
                expr_span: Span,
                is_range_bound: bool,
            ) {
                self.parser.dcx().try_steal_modify_and_emit_err(
                    stash_span,
                    StashKey::ExprInPat,
                    |err| {
                        // Includes pre-pats (e.g. `&mut <err>`) in the diagnostic.
                        err.span.replace(stash_span, expr_span);

                        let sm = self.parser.psess.source_map();
                        let stmt = self.stmt;
                        let line_lo = sm.span_extend_to_line(stmt.span).shrink_to_lo();
                        let indentation = sm.indentation_before(stmt.span).unwrap_or_default();
                        let Ok(expr) = self.parser.span_to_snippet(expr_span) else {
                            // FIXME: some suggestions don't actually need the snippet; see PR #123877's unresolved conversations.
                            return;
                        };

                        if let StmtKind::Let(local) = &stmt.kind {
                            match &local.kind {
                                LocalKind::Decl | LocalKind::Init(_) => {
                                    // It's kinda hard to guess what the user intended, so don't make suggestions.
                                    return;
                                }

                                LocalKind::InitElse(_, _) => {}
                            }
                        }

                        // help: use an arm guard `if val == expr`
                        // FIXME(guard_patterns): suggest this regardless of a match arm.
                        if let Some(arm) = &self.arm
                            && !is_range_bound
                        {
                            let (ident, ident_span) = match self.field {
                                Some(field) => {
                                    (field.ident.to_string(), field.ident.span.to(expr_span))
                                }
                                None => ("val".to_owned(), expr_span),
                            };

                            // Are parentheses required around `expr`?
                            // HACK: a neater way would be preferable.
                            let expr = match &err.args["expr_precedence"] {
                                DiagArgValue::Number(expr_precedence) => {
                                    if *expr_precedence <= ExprPrecedence::Compare as i32 {
                                        format!("({expr})")
                                    } else {
                                        format!("{expr}")
                                    }
                                }
                                _ => unreachable!(),
                            };

                            match &arm.guard {
                                None => {
                                    err.subdiagnostic(
                                        UnexpectedExpressionInPatternSugg::CreateGuard {
                                            ident_span,
                                            pat_hi: arm.pat.span.shrink_to_hi(),
                                            ident,
                                            expr,
                                        },
                                    );
                                }
                                Some(guard) => {
                                    // Are parentheses required around the old guard?
                                    let wrap_guard = guard.precedence() <= ExprPrecedence::LAnd;

                                    err.subdiagnostic(
                                        UnexpectedExpressionInPatternSugg::UpdateGuard {
                                            ident_span,
                                            guard_lo: if wrap_guard {
                                                Some(guard.span.shrink_to_lo())
                                            } else {
                                                None
                                            },
                                            guard_hi: guard.span.shrink_to_hi(),
                                            guard_hi_paren: if wrap_guard { ")" } else { "" },
                                            ident,
                                            expr,
                                        },
                                    );
                                }
                            }
                        }

                        // help: extract the expr into a `const VAL: _ = expr`
                        let ident = match self.field {
                            Some(field) => field.ident.as_str().to_uppercase(),
                            None => "VAL".to_owned(),
                        };
                        err.subdiagnostic(UnexpectedExpressionInPatternSugg::Const {
                            stmt_lo: line_lo,
                            ident_span: expr_span,
                            expr,
                            ident,
                            indentation,
                        });
                    },
                );
            }
        }

        impl<'a> Visitor<'a> for PatVisitor<'a> {
            fn visit_arm(&mut self, a: &'a Arm) -> Self::Result {
                self.arm = Some(a);
                visit::walk_arm(self, a);
                self.arm = None;
            }

            fn visit_pat_field(&mut self, fp: &'a PatField) -> Self::Result {
                self.field = Some(fp);
                visit::walk_pat_field(self, fp);
                self.field = None;
            }

            fn visit_pat(&mut self, p: &'a Pat) -> Self::Result {
                match &p.kind {
                    // Base expression
                    PatKind::Err(_) | PatKind::Expr(_) => {
                        self.maybe_add_suggestions_then_emit(p.span, p.span, false)
                    }

                    // Sub-patterns
                    // FIXME: this doesn't work with recursive subpats (`&mut &mut <err>`)
                    PatKind::Box(subpat) | PatKind::Ref(subpat, _)
                        if matches!(subpat.kind, PatKind::Err(_) | PatKind::Expr(_)) =>
                    {
                        self.maybe_add_suggestions_then_emit(subpat.span, p.span, false)
                    }

                    // Sub-expressions
                    PatKind::Range(start, end, _) => {
                        if let Some(start) = start {
                            self.maybe_add_suggestions_then_emit(start.span, start.span, true);
                        }

                        if let Some(end) = end {
                            self.maybe_add_suggestions_then_emit(end.span, end.span, true);
                        }
                    }

                    // Walk continuation
                    _ => visit::walk_pat(self, p),
                }
            }
        }

        // Starts the visit.
        PatVisitor { parser: self, stmt, arm: None, field: None }.visit_stmt(stmt);
    }

    fn eat_metavar_pat(&mut self) -> Option<P<Pat>> {
        // Must try both kinds of pattern nonterminals.
        if let Some(pat) = self.eat_metavar_seq_with_matcher(
            |mv_kind| matches!(mv_kind, MetaVarKind::Pat(PatParam { .. })),
            |this| this.parse_pat_no_top_alt(None, None),
        ) {
            Some(pat)
        } else if let Some(pat) = self.eat_metavar_seq(MetaVarKind::Pat(PatWithOr), |this| {
            this.parse_pat_no_top_guard(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::EitherTupleOrPipe,
            )
        }) {
            Some(pat)
        } else {
            None
        }
    }

    /// Parses a pattern, with a setting whether modern range patterns (e.g., `a..=b`, `a..b` are
    /// allowed).
    fn parse_pat_with_range_pat(
        &mut self,
        allow_range_pat: bool,
        expected: Option<Expected>,
        syntax_loc: Option<PatternLocation>,
    ) -> PResult<'a, P<Pat>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);

        if let Some(pat) = self.eat_metavar_pat() {
            return Ok(pat);
        }

        let mut lo = self.token.span;

        if self.token.is_keyword(kw::Let)
            && self.look_ahead(1, |tok| {
                tok.can_begin_pattern(token::NtPatKind::PatParam { inferred: false })
            })
        {
            self.bump();
            // Trim extra space after the `let`
            let span = lo.with_hi(self.token.span.lo());
            self.dcx().emit_err(RemoveLet { span: lo, suggestion: span });
            lo = self.token.span;
        }

        let pat = if self.check(exp!(And)) || self.token == token::AndAnd {
            self.parse_pat_deref(expected)?
        } else if self.check(exp!(OpenParen)) {
            self.parse_pat_tuple_or_parens()?
        } else if self.check(exp!(OpenBracket)) {
            // Parse `[pat, pat,...]` as a slice pattern.
            let (pats, _) =
                self.parse_delim_comma_seq(exp!(OpenBracket), exp!(CloseBracket), |p| {
                    p.parse_pat_allow_top_guard(
                        None,
                        RecoverComma::No,
                        RecoverColon::No,
                        CommaRecoveryMode::EitherTupleOrPipe,
                    )
                })?;
            PatKind::Slice(pats)
        } else if self.check(exp!(DotDot)) && !self.is_pat_range_end_start(1) {
            // A rest pattern `..`.
            self.bump(); // `..`
            PatKind::Rest
        } else if self.check(exp!(DotDotDot)) && !self.is_pat_range_end_start(1) {
            self.recover_dotdotdot_rest_pat(lo)
        } else if let Some(form) = self.parse_range_end() {
            self.parse_pat_range_to(form)? // `..=X`, `...X`, or `..X`.
        } else if self.eat(exp!(Bang)) {
            // Parse `!`
            self.psess.gated_spans.gate(sym::never_patterns, self.prev_token.span);
            PatKind::Never
        } else if self.eat_keyword(exp!(Underscore)) {
            // Parse `_`
            PatKind::Wild
        } else if self.eat_keyword(exp!(Mut)) {
            self.parse_pat_ident_mut()?
        } else if self.eat_keyword(exp!(Ref)) {
            if self.check_keyword(exp!(Box)) {
                // Suggest `box ref`.
                let span = self.prev_token.span.to(self.token.span);
                self.bump();
                self.dcx().emit_err(SwitchRefBoxOrder { span });
            }
            // Parse ref ident @ pat / ref mut ident @ pat
            let mutbl = self.parse_mutability();
            self.parse_pat_ident(BindingMode(ByRef::Yes(mutbl), Mutability::Not), syntax_loc)?
        } else if self.eat_keyword(exp!(Box)) {
            self.parse_pat_box()?
        } else if self.check_inline_const(0) {
            // Parse `const pat`
            let const_expr = self.parse_const_block(lo.to(self.token.span), true)?;

            if let Some(re) = self.parse_range_end() {
                self.parse_pat_range_begin_with(const_expr, re)?
            } else {
                PatKind::Expr(const_expr)
            }
        } else if self.is_builtin() {
            self.parse_pat_builtin()?
        }
        // Don't eagerly error on semantically invalid tokens when matching
        // declarative macros, as the input to those doesn't have to be
        // semantically valid. For attribute/derive proc macros this is not the
        // case, so doing the recovery for them is fine.
        else if self.can_be_ident_pat()
            || (self.is_lit_bad_ident().is_some() && self.may_recover())
        {
            // Parse `ident @ pat`
            // This can give false positives and parse nullary enums,
            // they are dealt with later in resolve.
            self.parse_pat_ident(BindingMode::NONE, syntax_loc)?
        } else if self.is_start_of_pat_with_path() {
            // Parse pattern starting with a path
            let (qself, path) = if self.eat_lt() {
                // Parse a qualified path
                let (qself, path) = self.parse_qpath(PathStyle::Pat)?;
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, self.parse_path(PathStyle::Pat)?)
            };
            let span = lo.to(self.prev_token.span);

            if qself.is_none() && self.check(exp!(Bang)) {
                self.parse_pat_mac_invoc(path)?
            } else if let Some(form) = self.parse_range_end() {
                let begin = self.mk_expr(span, ExprKind::Path(qself, path));
                self.parse_pat_range_begin_with(begin, form)?
            } else if self.check(exp!(OpenBrace)) {
                self.parse_pat_struct(qself, path)?
            } else if self.check(exp!(OpenParen)) {
                self.parse_pat_tuple_struct(qself, path)?
            } else {
                match self.maybe_recover_trailing_expr(span, false) {
                    Some((guar, _)) => PatKind::Err(guar),
                    None => PatKind::Path(qself, path),
                }
            }
        } else if let Some((lt, IdentIsRaw::No)) = self.token.lifetime()
            // In pattern position, we're totally fine with using "next token isn't colon"
            // as a heuristic. We could probably just always try to recover if it's a lifetime,
            // because we never have `'a: label {}` in a pattern position anyways, but it does
            // keep us from suggesting something like `let 'a: Ty = ..` => `let 'a': Ty = ..`
            && could_be_unclosed_char_literal(lt)
            && !self.look_ahead(1, |token| token.kind == token::Colon)
        {
            // Recover a `'a` as a `'a'` literal
            let lt = self.expect_lifetime();
            let (lit, _) =
                self.recover_unclosed_char(lt.ident, Parser::mk_token_lit_char, |self_| {
                    let expected = Expected::to_string_or_fallback(expected);
                    let msg = format!(
                        "expected {}, found {}",
                        expected,
                        super::token_descr(&self_.token)
                    );

                    self_
                        .dcx()
                        .struct_span_err(self_.token.span, msg)
                        .with_span_label(self_.token.span, format!("expected {expected}"))
                });
            PatKind::Expr(self.mk_expr(lo, ExprKind::Lit(lit)))
        } else {
            // Try to parse everything else as literal with optional minus
            match self.parse_literal_maybe_minus() {
                Ok(begin) => {
                    let begin = self
                        .maybe_recover_trailing_expr(begin.span, false)
                        .map(|(guar, sp)| self.mk_expr_err(sp, guar))
                        .unwrap_or(begin);

                    match self.parse_range_end() {
                        Some(form) => self.parse_pat_range_begin_with(begin, form)?,
                        None => PatKind::Expr(begin),
                    }
                }
                Err(err) => return self.fatal_unexpected_non_pat(err, expected),
            }
        };

        let pat = self.mk_pat(lo.to(self.prev_token.span), pat);
        let pat = self.maybe_recover_from_bad_qpath(pat)?;
        let pat = self.recover_intersection_pat(pat)?;

        if !allow_range_pat {
            self.ban_pat_range_if_ambiguous(&pat)
        }

        Ok(pat)
    }

    /// Recover from a typoed `...` pattern that was encountered
    /// Ref: Issue #70388
    fn recover_dotdotdot_rest_pat(&mut self, lo: Span) -> PatKind {
        // A typoed rest pattern `...`.
        self.bump(); // `...`

        // The user probably mistook `...` for a rest pattern `..`.
        self.dcx().emit_err(DotDotDotRestPattern {
            span: lo,
            suggestion: lo.with_lo(lo.hi() - BytePos(1)),
        });
        PatKind::Rest
    }

    /// Try to recover the more general form `intersect ::= $pat_lhs @ $pat_rhs`.
    ///
    /// Allowed binding patterns generated by `binding ::= ref? mut? $ident @ $pat_rhs`
    /// should already have been parsed by now at this point,
    /// if the next token is `@` then we can try to parse the more general form.
    ///
    /// Consult `parse_pat_ident` for the `binding` grammar.
    ///
    /// The notion of intersection patterns are found in
    /// e.g. [F#][and] where they are called AND-patterns.
    ///
    /// [and]: https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/pattern-matching
    fn recover_intersection_pat(&mut self, lhs: P<Pat>) -> PResult<'a, P<Pat>> {
        if self.token != token::At {
            // Next token is not `@` so it's not going to be an intersection pattern.
            return Ok(lhs);
        }

        // At this point we attempt to parse `@ $pat_rhs` and emit an error.
        self.bump(); // `@`
        let mut rhs = self.parse_pat_no_top_alt(None, None)?;
        let whole_span = lhs.span.to(rhs.span);

        if let PatKind::Ident(_, _, sub @ None) = &mut rhs.kind {
            // The user inverted the order, so help them fix that.
            let lhs_span = lhs.span;
            // Move the LHS into the RHS as a subpattern.
            // The RHS is now the full pattern.
            *sub = Some(lhs);

            self.dcx().emit_err(PatternOnWrongSideOfAt {
                whole_span,
                whole_pat: pprust::pat_to_string(&rhs),
                pattern: lhs_span,
                binding: rhs.span,
            });
        } else {
            // The special case above doesn't apply so we may have e.g. `A(x) @ B(y)`.
            rhs.kind = PatKind::Wild;
            self.dcx().emit_err(ExpectedBindingLeftOfAt {
                whole_span,
                lhs: lhs.span,
                rhs: rhs.span,
            });
        }

        rhs.span = whole_span;
        Ok(rhs)
    }

    /// Ban a range pattern if it has an ambiguous interpretation.
    fn ban_pat_range_if_ambiguous(&self, pat: &Pat) {
        match pat.kind {
            PatKind::Range(
                ..,
                Spanned { node: RangeEnd::Included(RangeSyntax::DotDotDot), .. },
            ) => return,
            PatKind::Range(..) => {}
            _ => return,
        }

        self.dcx().emit_err(AmbiguousRangePattern {
            span: pat.span,
            suggestion: ParenRangeSuggestion {
                lo: pat.span.shrink_to_lo(),
                hi: pat.span.shrink_to_hi(),
            },
        });
    }

    /// Parse `&pat` / `&mut pat`.
    fn parse_pat_deref(&mut self, expected: Option<Expected>) -> PResult<'a, PatKind> {
        self.expect_and()?;
        if let Some((lifetime, _)) = self.token.lifetime() {
            self.bump(); // `'a`

            self.dcx().emit_err(UnexpectedLifetimeInPattern {
                span: self.prev_token.span,
                symbol: lifetime.name,
                suggestion: self.prev_token.span.until(self.token.span),
            });
        }

        let mutbl = self.parse_mutability();
        let subpat = self.parse_pat_with_range_pat(false, expected, None)?;
        Ok(PatKind::Ref(subpat, mutbl))
    }

    /// Parse a tuple or parenthesis pattern.
    fn parse_pat_tuple_or_parens(&mut self) -> PResult<'a, PatKind> {
        let open_paren = self.token.span;

        let (fields, trailing_comma) = self.parse_paren_comma_seq(|p| {
            p.parse_pat_allow_top_guard(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::LikelyTuple,
            )
        })?;

        // Here, `(pat,)` is a tuple pattern.
        // For backward compatibility, `(..)` is a tuple pattern as well.
        let paren_pattern =
            fields.len() == 1 && !(matches!(trailing_comma, Trailing::Yes) || fields[0].is_rest());

        let pat = if paren_pattern {
            let pat = fields.into_iter().next().unwrap();
            let close_paren = self.prev_token.span;

            match &pat.kind {
                // recover ranges with parentheses around the `(start)..`
                PatKind::Expr(begin)
                    if self.may_recover()
                        && let Some(form) = self.parse_range_end() =>
                {
                    self.dcx().emit_err(UnexpectedParenInRangePat {
                        span: vec![open_paren, close_paren],
                        sugg: UnexpectedParenInRangePatSugg {
                            start_span: open_paren,
                            end_span: close_paren,
                        },
                    });

                    self.parse_pat_range_begin_with(begin.clone(), form)?
                }
                // recover ranges with parentheses around the `(start)..`
                PatKind::Err(guar)
                    if self.may_recover()
                        && let Some(form) = self.parse_range_end() =>
                {
                    self.dcx().emit_err(UnexpectedParenInRangePat {
                        span: vec![open_paren, close_paren],
                        sugg: UnexpectedParenInRangePatSugg {
                            start_span: open_paren,
                            end_span: close_paren,
                        },
                    });

                    self.parse_pat_range_begin_with(self.mk_expr_err(pat.span, *guar), form)?
                }

                // (pat) with optional parentheses
                _ => PatKind::Paren(pat),
            }
        } else {
            PatKind::Tuple(fields)
        };

        Ok(match self.maybe_recover_trailing_expr(open_paren.to(self.prev_token.span), false) {
            None => pat,
            Some((guar, _)) => PatKind::Err(guar),
        })
    }

    /// Parse a mutable binding with the `mut` token already eaten.
    fn parse_pat_ident_mut(&mut self) -> PResult<'a, PatKind> {
        let mut_span = self.prev_token.span;

        self.recover_additional_muts();

        let byref = self.parse_byref();

        self.recover_additional_muts();

        // Make sure we don't allow e.g. `let mut $p;` where `$p:pat`.
        if let Some(MetaVarKind::Pat(_)) = self.token.is_metavar_seq() {
            self.expected_ident_found_err().emit();
        }

        // Parse the pattern we hope to be an identifier.
        let mut pat = self.parse_pat_no_top_alt(Some(Expected::Identifier), None)?;

        // If we don't have `mut $ident (@ pat)?`, error.
        if let PatKind::Ident(BindingMode(br @ ByRef::No, m @ Mutability::Not), ..) = &mut pat.kind
        {
            // Don't recurse into the subpattern.
            // `mut` on the outer binding doesn't affect the inner bindings.
            *br = byref;
            *m = Mutability::Mut;
        } else {
            // Add `mut` to any binding in the parsed pattern.
            let changed_any_binding = Self::make_all_value_bindings_mutable(&mut pat);
            self.ban_mut_general_pat(mut_span, &pat, changed_any_binding);
        }

        if matches!(pat.kind, PatKind::Ident(BindingMode(ByRef::Yes(_), Mutability::Mut), ..)) {
            self.psess.gated_spans.gate(sym::mut_ref, pat.span);
        }
        Ok(pat.into_inner().kind)
    }

    /// Turn all by-value immutable bindings in a pattern into mutable bindings.
    /// Returns `true` if any change was made.
    fn make_all_value_bindings_mutable(pat: &mut P<Pat>) -> bool {
        struct AddMut(bool);
        impl MutVisitor for AddMut {
            fn visit_pat(&mut self, pat: &mut P<Pat>) {
                if let PatKind::Ident(BindingMode(ByRef::No, m @ Mutability::Not), ..) =
                    &mut pat.kind
                {
                    self.0 = true;
                    *m = Mutability::Mut;
                }
                mut_visit::walk_pat(self, pat);
            }
        }

        let mut add_mut = AddMut(false);
        add_mut.visit_pat(pat);
        add_mut.0
    }

    /// Error on `mut $pat` where `$pat` is not an ident.
    fn ban_mut_general_pat(&self, lo: Span, pat: &Pat, changed_any_binding: bool) {
        self.dcx().emit_err(if changed_any_binding {
            InvalidMutInPattern::NestedIdent {
                span: lo.to(pat.span),
                pat: pprust::pat_to_string(pat),
            }
        } else {
            InvalidMutInPattern::NonIdent { span: lo.until(pat.span) }
        });
    }

    /// Eat any extraneous `mut`s and error + recover if we ate any.
    fn recover_additional_muts(&mut self) {
        let lo = self.token.span;
        while self.eat_keyword(exp!(Mut)) {}
        if lo == self.token.span {
            return;
        }

        let span = lo.to(self.prev_token.span);
        let suggestion = span.with_hi(self.token.span.lo());
        self.dcx().emit_err(RepeatedMutInPattern { span, suggestion });
    }

    /// Parse macro invocation
    fn parse_pat_mac_invoc(&mut self, path: Path) -> PResult<'a, PatKind> {
        self.bump();
        let args = self.parse_delim_args()?;
        let mac = P(MacCall { path, args });
        Ok(PatKind::MacCall(mac))
    }

    fn fatal_unexpected_non_pat(
        &mut self,
        err: Diag<'a>,
        expected: Option<Expected>,
    ) -> PResult<'a, P<Pat>> {
        err.cancel();

        let expected = Expected::to_string_or_fallback(expected);
        let msg = format!("expected {}, found {}", expected, super::token_descr(&self.token));

        let mut err = self.dcx().struct_span_err(self.token.span, msg);
        err.span_label(self.token.span, format!("expected {expected}"));

        let sp = self.psess.source_map().start_point(self.token.span);
        if let Some(sp) = self.psess.ambiguous_block_expr_parse.borrow().get(&sp) {
            err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
        }

        Err(err)
    }

    /// Parses the range pattern end form `".." | "..." | "..=" ;`.
    fn parse_range_end(&mut self) -> Option<Spanned<RangeEnd>> {
        let re = if self.eat(exp!(DotDotDot)) {
            RangeEnd::Included(RangeSyntax::DotDotDot)
        } else if self.eat(exp!(DotDotEq)) {
            RangeEnd::Included(RangeSyntax::DotDotEq)
        } else if self.eat(exp!(DotDot)) {
            RangeEnd::Excluded
        } else {
            return None;
        };
        Some(respan(self.prev_token.span, re))
    }

    /// Parse a range pattern `$begin $form $end?` where `$form = ".." | "..." | "..=" ;`.
    /// `$begin $form` has already been parsed.
    fn parse_pat_range_begin_with(
        &mut self,
        begin: P<Expr>,
        re: Spanned<RangeEnd>,
    ) -> PResult<'a, PatKind> {
        let end = if self.is_pat_range_end_start(0) {
            // Parsing e.g. `X..=Y`.
            Some(self.parse_pat_range_end()?)
        } else {
            // Parsing e.g. `X..`.
            if let RangeEnd::Included(_) = re.node {
                // FIXME(Centril): Consider semantic errors instead in `ast_validation`.
                self.inclusive_range_with_incorrect_end();
            }
            None
        };
        Ok(PatKind::Range(Some(begin), end, re))
    }

    pub(super) fn inclusive_range_with_incorrect_end(&mut self) -> ErrorGuaranteed {
        let tok = &self.token;
        let span = self.prev_token.span;
        // If the user typed "..==" instead of "..=", we want to give them
        // a specific error message telling them to use "..=".
        // If they typed "..=>", suggest they use ".. =>".
        // Otherwise, we assume that they meant to type a half open exclusive
        // range and give them an error telling them to do that instead.
        let no_space = tok.span.lo() == span.hi();
        match tok.kind {
            token::Eq if no_space => {
                let span_with_eq = span.to(tok.span);

                // Ensure the user doesn't receive unhelpful unexpected token errors
                self.bump();
                if self.is_pat_range_end_start(0) {
                    let _ = self.parse_pat_range_end().map_err(|e| e.cancel());
                }

                self.dcx().emit_err(InclusiveRangeExtraEquals { span: span_with_eq })
            }
            token::Gt if no_space => {
                let after_pat = span.with_hi(span.hi() - BytePos(1)).shrink_to_hi();
                self.dcx().emit_err(InclusiveRangeMatchArrow { span, arrow: tok.span, after_pat })
            }
            _ => self.dcx().emit_err(InclusiveRangeNoEnd {
                span,
                suggestion: span.with_lo(span.hi() - BytePos(1)),
            }),
        }
    }

    /// Parse a range-to pattern, `..X` or `..=X` where `X` remains to be parsed.
    ///
    /// The form `...X` is prohibited to reduce confusion with the potential
    /// expression syntax `...expr` for splatting in expressions.
    fn parse_pat_range_to(&mut self, mut re: Spanned<RangeEnd>) -> PResult<'a, PatKind> {
        let end = self.parse_pat_range_end()?;
        if let RangeEnd::Included(syn @ RangeSyntax::DotDotDot) = &mut re.node {
            *syn = RangeSyntax::DotDotEq;
            self.dcx().emit_err(DotDotDotRangeToPatternNotAllowed { span: re.span });
        }
        Ok(PatKind::Range(None, Some(end), re))
    }

    /// Is the token `dist` away from the current suitable as the start of a range patterns end?
    fn is_pat_range_end_start(&self, dist: usize) -> bool {
        self.check_inline_const(dist)
            || self.look_ahead(dist, |t| {
                t.is_path_start() // e.g. `MY_CONST`;
                || *t == token::Dot // e.g. `.5` for recovery;
                || matches!(t.kind, token::Literal(..) | token::Minus)
                || t.is_bool_lit()
                || t.is_metavar_expr()
                || t.is_lifetime() // recover `'a` instead of `'a'`
                || (self.may_recover() // recover leading `(`
                    && *t == token::OpenParen
                    && self.look_ahead(dist + 1, |t| *t != token::OpenParen)
                    && self.is_pat_range_end_start(dist + 1))
            })
    }

    /// Parse a range pattern end bound
    fn parse_pat_range_end(&mut self) -> PResult<'a, P<Expr>> {
        // recover leading `(`
        let open_paren = (self.may_recover() && self.eat_noexpect(&token::OpenParen))
            .then_some(self.prev_token.span);

        let bound = if self.check_inline_const(0) {
            self.parse_const_block(self.token.span, true)
        } else if self.check_path() {
            let lo = self.token.span;
            let (qself, path) = if self.eat_lt() {
                // Parse a qualified path
                let (qself, path) = self.parse_qpath(PathStyle::Pat)?;
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, self.parse_path(PathStyle::Pat)?)
            };
            let hi = self.prev_token.span;
            Ok(self.mk_expr(lo.to(hi), ExprKind::Path(qself, path)))
        } else {
            self.parse_literal_maybe_minus()
        }?;

        let recovered = self.maybe_recover_trailing_expr(bound.span, true);

        // recover trailing `)`
        if let Some(open_paren) = open_paren {
            self.expect(exp!(CloseParen))?;

            self.dcx().emit_err(UnexpectedParenInRangePat {
                span: vec![open_paren, self.prev_token.span],
                sugg: UnexpectedParenInRangePatSugg {
                    start_span: open_paren,
                    end_span: self.prev_token.span,
                },
            });
        }

        Ok(match recovered {
            Some((guar, sp)) => self.mk_expr_err(sp, guar),
            None => bound,
        })
    }

    /// Is this the start of a pattern beginning with a path?
    fn is_start_of_pat_with_path(&mut self) -> bool {
        self.check_path()
        // Just for recovery (see `can_be_ident`).
        || self.token.is_ident() && !self.token.is_bool_lit() && !self.token.is_keyword(kw::In)
    }

    /// Would `parse_pat_ident` be appropriate here?
    fn can_be_ident_pat(&mut self) -> bool {
        self.check_ident()
        && !self.token.is_bool_lit() // Avoid `true` or `false` as a binding as it is a literal.
        && !self.token.is_path_segment_keyword() // Avoid e.g. `Self` as it is a path.
        // Avoid `in`. Due to recovery in the list parser this messes with `for ( $pat in $expr )`.
        && !self.token.is_keyword(kw::In)
        // Try to do something more complex?
        && self.look_ahead(1, |t| !matches!(t.kind, token::OpenParen // A tuple struct pattern.
            | token::OpenBrace // A struct pattern.
            | token::DotDotDot | token::DotDotEq | token::DotDot // A range pattern.
            | token::PathSep // A tuple / struct variant pattern.
            | token::Bang)) // A macro expanding to a pattern.
    }

    /// Parses `ident` or `ident @ pat`.
    /// Used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like `ref foo(a, b)`.
    fn parse_pat_ident(
        &mut self,
        binding_annotation: BindingMode,
        syntax_loc: Option<PatternLocation>,
    ) -> PResult<'a, PatKind> {
        let ident = self.parse_ident_common(false)?;

        if self.may_recover()
            && !matches!(syntax_loc, Some(PatternLocation::FunctionParameter))
            && self.check_noexpect(&token::Lt)
            && self.look_ahead(1, |t| t.can_begin_type())
        {
            return Err(self.dcx().create_err(GenericArgsInPatRequireTurbofishSyntax {
                span: self.token.span,
                suggest_turbofish: self.token.span.shrink_to_lo(),
            }));
        }

        let sub = if self.eat(exp!(At)) {
            Some(self.parse_pat_no_top_alt(Some(Expected::BindingPattern), None)?)
        } else {
            None
        };

        // Just to be friendly, if they write something like `ref Some(i)`,
        // we end up here with `(` as the current token.
        // This shortly leads to a parse error. Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to `parse_enum_variant()`.
        if self.token == token::OpenParen {
            return Err(self
                .dcx()
                .create_err(EnumPatternInsteadOfIdentifier { span: self.prev_token.span }));
        }

        // Check for method calls after the `ident`,
        // but not `ident @ subpat` as `subpat` was already checked and `ident` continues with `@`.

        let pat = if sub.is_none()
            && let Some((guar, _)) = self.maybe_recover_trailing_expr(ident.span, false)
        {
            PatKind::Err(guar)
        } else {
            PatKind::Ident(binding_annotation, ident, sub)
        };
        Ok(pat)
    }

    /// Parse a struct ("record") pattern (e.g. `Foo { ... }` or `Foo::Bar { ... }`).
    fn parse_pat_struct(&mut self, qself: Option<P<QSelf>>, path: Path) -> PResult<'a, PatKind> {
        if qself.is_some() {
            // Feature gate the use of qualified paths in patterns
            self.psess.gated_spans.gate(sym::more_qualified_paths, path.span);
        }
        self.bump();
        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
            e.span_label(path.span, "while parsing the fields for this pattern");
            let guar = e.emit();
            self.recover_stmt();
            // When recovering, pretend we had `Foo { .. }`, to avoid cascading errors.
            (ThinVec::new(), PatFieldsRest::Recovered(guar))
        });
        self.bump();
        Ok(PatKind::Struct(qself, path, fields, etc))
    }

    /// Parse tuple struct or tuple variant pattern (e.g. `Foo(...)` or `Foo::Bar(...)`).
    fn parse_pat_tuple_struct(
        &mut self,
        qself: Option<P<QSelf>>,
        path: Path,
    ) -> PResult<'a, PatKind> {
        let (fields, _) = self.parse_paren_comma_seq(|p| {
            p.parse_pat_allow_top_guard(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::EitherTupleOrPipe,
            )
        })?;
        if qself.is_some() {
            self.psess.gated_spans.gate(sym::more_qualified_paths, path.span);
        }
        Ok(PatKind::TupleStruct(qself, path, fields))
    }

    /// Are we sure this could not possibly be the start of a pattern?
    ///
    /// Currently, this only accounts for tokens that can follow identifiers
    /// in patterns, but this can be extended as necessary.
    fn isnt_pattern_start(&self) -> bool {
        [
            token::Eq,
            token::Colon,
            token::Comma,
            token::Semi,
            token::At,
            token::OpenBrace,
            token::CloseBrace,
            token::CloseParen,
        ]
        .contains(&self.token.kind)
    }

    fn parse_pat_builtin(&mut self) -> PResult<'a, PatKind> {
        self.parse_builtin(|self_, _lo, ident| {
            Ok(match ident.name {
                // builtin#deref(PAT)
                sym::deref => Some(ast::PatKind::Deref(self_.parse_pat_allow_top_guard(
                    None,
                    RecoverComma::Yes,
                    RecoverColon::Yes,
                    CommaRecoveryMode::LikelyTuple,
                )?)),
                _ => None,
            })
        })
    }

    /// Parses `box pat`
    fn parse_pat_box(&mut self) -> PResult<'a, PatKind> {
        let box_span = self.prev_token.span;

        if self.isnt_pattern_start() {
            let descr = super::token_descr(&self.token);
            self.dcx().emit_err(errors::BoxNotPat {
                span: self.token.span,
                kw: box_span,
                lo: box_span.shrink_to_lo(),
                descr,
            });

            // We cannot use `parse_pat_ident()` since it will complain `box`
            // is not an identifier.
            let sub = if self.eat(exp!(At)) {
                Some(self.parse_pat_no_top_alt(Some(Expected::BindingPattern), None)?)
            } else {
                None
            };

            Ok(PatKind::Ident(BindingMode::NONE, Ident::new(kw::Box, box_span), sub))
        } else {
            let pat = self.parse_pat_with_range_pat(false, None, None)?;
            self.psess.gated_spans.gate(sym::box_patterns, box_span.to(self.prev_token.span));
            Ok(PatKind::Box(pat))
        }
    }

    /// Parses the fields of a struct-like pattern.
    fn parse_pat_fields(&mut self) -> PResult<'a, (ThinVec<PatField>, PatFieldsRest)> {
        let mut fields: ThinVec<PatField> = ThinVec::new();
        let mut etc = PatFieldsRest::None;
        let mut ate_comma = true;
        let mut delayed_err: Option<Diag<'a>> = None;
        let mut first_etc_and_maybe_comma_span = None;
        let mut last_non_comma_dotdot_span = None;

        while self.token != token::CloseBrace {
            // check that a comma comes after every field
            if !ate_comma {
                let err = if self.token == token::At {
                    let prev_field = fields
                        .last()
                        .expect("Unreachable on first iteration, not empty otherwise")
                        .ident;
                    self.report_misplaced_at_in_struct_pat(prev_field)
                } else {
                    let mut err = self
                        .dcx()
                        .create_err(ExpectedCommaAfterPatternField { span: self.token.span });
                    self.recover_misplaced_pattern_modifiers(&fields, &mut err);
                    err
                };
                if let Some(delayed) = delayed_err {
                    delayed.emit();
                }
                return Err(err);
            }
            ate_comma = false;

            if self.check(exp!(DotDot))
                || self.check_noexpect(&token::DotDotDot)
                || self.check_keyword(exp!(Underscore))
            {
                etc = PatFieldsRest::Rest;
                let mut etc_sp = self.token.span;
                if first_etc_and_maybe_comma_span.is_none() {
                    if let Some(comma_tok) =
                        self.look_ahead(1, |&t| if t == token::Comma { Some(t) } else { None })
                    {
                        let nw_span = self
                            .psess
                            .source_map()
                            .span_extend_to_line(comma_tok.span)
                            .trim_start(comma_tok.span.shrink_to_lo())
                            .map(|s| self.psess.source_map().span_until_non_whitespace(s));
                        first_etc_and_maybe_comma_span = nw_span.map(|s| etc_sp.to(s));
                    } else {
                        first_etc_and_maybe_comma_span =
                            Some(self.psess.source_map().span_until_non_whitespace(etc_sp));
                    }
                }

                self.recover_bad_dot_dot();
                self.bump(); // `..` || `...` || `_`

                if self.token == token::CloseBrace {
                    break;
                }
                let token_str = super::token_descr(&self.token);
                let msg = format!("expected `}}`, found {token_str}");
                let mut err = self.dcx().struct_span_err(self.token.span, msg);

                err.span_label(self.token.span, "expected `}`");
                let mut comma_sp = None;
                if self.token == token::Comma {
                    // Issue #49257
                    let nw_span =
                        self.psess.source_map().span_until_non_whitespace(self.token.span);
                    etc_sp = etc_sp.to(nw_span);
                    err.span_label(
                        etc_sp,
                        "`..` must be at the end and cannot have a trailing comma",
                    );
                    comma_sp = Some(self.token.span);
                    self.bump();
                    ate_comma = true;
                }

                if self.token == token::CloseBrace {
                    // If the struct looks otherwise well formed, recover and continue.
                    if let Some(sp) = comma_sp {
                        err.span_suggestion_short(
                            sp,
                            "remove this comma",
                            "",
                            Applicability::MachineApplicable,
                        );
                    }
                    err.emit();
                    break;
                } else if self.token.is_ident() && ate_comma {
                    // Accept fields coming after `..,`.
                    // This way we avoid "pattern missing fields" errors afterwards.
                    // We delay this error until the end in order to have a span for a
                    // suggested fix.
                    if let Some(delayed_err) = delayed_err {
                        delayed_err.emit();
                        return Err(err);
                    } else {
                        delayed_err = Some(err);
                    }
                } else {
                    if let Some(err) = delayed_err {
                        err.emit();
                    }
                    return Err(err);
                }
            }

            let attrs = match self.parse_outer_attributes() {
                Ok(attrs) => attrs,
                Err(err) => {
                    if let Some(delayed) = delayed_err {
                        delayed.emit();
                    }
                    return Err(err);
                }
            };
            let lo = self.token.span;

            let field = self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
                let field = match this.parse_pat_field(lo, attrs) {
                    Ok(field) => Ok(field),
                    Err(err) => {
                        if let Some(delayed_err) = delayed_err.take() {
                            delayed_err.emit();
                        }
                        return Err(err);
                    }
                }?;
                ate_comma = this.eat(exp!(Comma));

                last_non_comma_dotdot_span = Some(this.prev_token.span);

                // We just ate a comma, so there's no need to capture a trailing token.
                Ok((field, Trailing::No, UsePreAttrPos::No))
            })?;

            fields.push(field)
        }

        if let Some(mut err) = delayed_err {
            if let Some(first_etc_span) = first_etc_and_maybe_comma_span {
                if self.prev_token == token::DotDot {
                    // We have `.., x, ..`.
                    err.multipart_suggestion(
                        "remove the starting `..`",
                        vec![(first_etc_span, String::new())],
                        Applicability::MachineApplicable,
                    );
                } else if let Some(last_non_comma_dotdot_span) = last_non_comma_dotdot_span {
                    // We have `.., x`.
                    err.multipart_suggestion(
                        "move the `..` to the end of the field list",
                        vec![
                            (first_etc_span, String::new()),
                            (
                                self.token.span.to(last_non_comma_dotdot_span.shrink_to_hi()),
                                format!("{} .. }}", if ate_comma { "" } else { "," }),
                            ),
                        ],
                        Applicability::MachineApplicable,
                    );
                }
            }
            err.emit();
        }
        Ok((fields, etc))
    }

    #[deny(rustc::untranslatable_diagnostic)]
    fn report_misplaced_at_in_struct_pat(&self, prev_field: Ident) -> Diag<'a> {
        debug_assert_eq!(self.token, token::At);
        let span = prev_field.span.to(self.token.span);
        if let Some(dot_dot_span) =
            self.look_ahead(1, |t| if t == &token::DotDot { Some(t.span) } else { None })
        {
            self.dcx().create_err(AtDotDotInStructPattern {
                span: span.to(dot_dot_span),
                remove: span.until(dot_dot_span),
                ident: prev_field,
            })
        } else {
            self.dcx().create_err(AtInStructPattern { span })
        }
    }

    /// If the user writes `S { ref field: name }` instead of `S { field: ref name }`, we suggest
    /// the correct code.
    fn recover_misplaced_pattern_modifiers(&self, fields: &ThinVec<PatField>, err: &mut Diag<'a>) {
        if let Some(last) = fields.iter().last()
            && last.is_shorthand
            && let PatKind::Ident(binding, ident, None) = last.pat.kind
            && binding != BindingMode::NONE
            && self.token == token::Colon
            // We found `ref mut? ident:`, try to parse a `name,` or `name }`.
            && let Some(name_span) = self.look_ahead(1, |t| t.is_ident().then(|| t.span))
            && self.look_ahead(2, |t| {
                t == &token::Comma || t == &token::CloseBrace
            })
        {
            let span = last.pat.span.with_hi(ident.span.lo());
            // We have `S { ref field: name }` instead of `S { field: ref name }`
            err.multipart_suggestion(
                "the pattern modifiers belong after the `:`",
                vec![
                    (span, String::new()),
                    (name_span.shrink_to_lo(), binding.prefix_str().to_string()),
                ],
                Applicability::MachineApplicable,
            );
        }
    }

    /// Recover on `...` or `_` as if it were `..` to avoid further errors.
    /// See issue #46718.
    fn recover_bad_dot_dot(&self) {
        if self.token == token::DotDot {
            return;
        }

        let token_str = pprust::token_to_string(&self.token);
        self.dcx().emit_err(DotDotDotForRemainingFields { span: self.token.span, token_str });
    }

    fn parse_pat_field(&mut self, lo: Span, attrs: AttrVec) -> PResult<'a, PatField> {
        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let hi;
        let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
            // Parsing a pattern of the form `fieldname: pat`.
            let fieldname = self.parse_field_name()?;
            self.bump();
            let pat = self.parse_pat_allow_top_guard(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::EitherTupleOrPipe,
            )?;
            hi = pat.span;
            (pat, fieldname, false)
        } else {
            // Parsing a pattern of the form `(box) (ref) (mut) fieldname`.
            let is_box = self.eat_keyword(exp!(Box));
            let boxed_span = self.token.span;
            let mutability = self.parse_mutability();
            let by_ref = self.parse_byref();

            let fieldname = self.parse_field_name()?;
            hi = self.prev_token.span;
            let ann = BindingMode(by_ref, mutability);
            let fieldpat = self.mk_pat_ident(boxed_span.to(hi), ann, fieldname);
            let subpat =
                if is_box { self.mk_pat(lo.to(hi), PatKind::Box(fieldpat)) } else { fieldpat };
            (subpat, fieldname, true)
        };

        Ok(PatField {
            ident: fieldname,
            pat: subpat,
            is_shorthand,
            attrs,
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            is_placeholder: false,
        })
    }

    pub(super) fn mk_pat_ident(&self, span: Span, ann: BindingMode, ident: Ident) -> P<Pat> {
        self.mk_pat(span, PatKind::Ident(ann, ident, None))
    }

    pub(super) fn mk_pat(&self, span: Span, kind: PatKind) -> P<Pat> {
        P(Pat { kind, span, id: ast::DUMMY_NODE_ID, tokens: None })
    }
}
