use super::{ForceCollect, Parser, PathStyle, TrailingToken};
use crate::errors::{
    self, AmbiguousRangePattern, DotDotDotForRemainingFields, DotDotDotRangeToPatternNotAllowed,
    DotDotDotRestPattern, EnumPatternInsteadOfIdentifier, ExpectedBindingLeftOfAt,
    ExpectedCommaAfterPatternField, InclusiveRangeExtraEquals, InclusiveRangeMatchArrow,
    InclusiveRangeNoEnd, InvalidMutInPattern, PatternOnWrongSideOfAt, RefMutOrderIncorrect,
    RemoveLet, RepeatedMutInPattern, TopLevelOrPatternNotAllowed, TopLevelOrPatternNotAllowedSugg,
    TrailingVertNotAllowed, UnexpectedLifetimeInPattern, UnexpectedVertVertBeforeFunctionParam,
    UnexpectedVertVertInPattern,
};
use crate::fluent_generated as fluent;
use crate::{maybe_recover_from_interpolated_ty_qpath, maybe_whole};
use rustc_ast::mut_visit::{noop_visit_pat, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter};
use rustc_ast::{
    self as ast, AttrVec, BindingAnnotation, ByRef, Expr, ExprKind, MacCall, Mutability, Pat,
    PatField, PatKind, Path, QSelf, RangeEnd, RangeSyntax,
};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, DiagnosticBuilder, ErrorGuaranteed, IntoDiagnostic, PResult};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::source_map::{respan, Span, Spanned};
use rustc_span::symbol::{kw, sym, Ident};
use thin_vec::{thin_vec, ThinVec};

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
pub(super) enum PatternLocation {
    LetBinding,
    FunctionParameter,
}

impl<'a> Parser<'a> {
    /// Parses a pattern.
    ///
    /// Corresponds to `pat<no_top_alt>` in RFC 2535 and does not admit or-patterns
    /// at the top level. Used when parsing the parameters of lambda expressions,
    /// functions, function pointers, and `pat` macro fragments.
    pub fn parse_pat_no_top_alt(&mut self, expected: Option<Expected>) -> PResult<'a, P<Pat>> {
        self.parse_pat_with_range_pat(true, expected)
    }

    /// Parses a pattern.
    ///
    /// Corresponds to `top_pat` in RFC 2535 and allows or-pattern at the top level.
    /// Used for parsing patterns in all cases when `pat<no_top_alt>` is not used.
    ///
    /// Note that after the FCP in <https://github.com/rust-lang/rust/issues/81415>,
    /// a leading vert is allowed in nested or-patterns, too. This allows us to
    /// simplify the grammar somewhat.
    pub fn parse_pat_allow_top_alt(
        &mut self,
        expected: Option<Expected>,
        rc: RecoverComma,
        ra: RecoverColon,
        rt: CommaRecoveryMode,
    ) -> PResult<'a, P<Pat>> {
        self.parse_pat_allow_top_alt_inner(expected, rc, ra, rt).map(|(pat, _)| pat)
    }

    /// Returns the pattern and a bool indicating whether we recovered from a trailing vert (true =
    /// recovered).
    fn parse_pat_allow_top_alt_inner(
        &mut self,
        expected: Option<Expected>,
        rc: RecoverComma,
        ra: RecoverColon,
        rt: CommaRecoveryMode,
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
        let mut first_pat = self.parse_pat_no_top_alt(expected)?;
        if rc == RecoverComma::Yes {
            self.maybe_recover_unexpected_comma(first_pat.span, rt)?;
        }

        // If the next token is not a `|`,
        // this is not an or-pattern and we should exit here.
        if !self.check(&token::BinOp(token::Or)) && self.token != token::OrOr {
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
            let pat = self.parse_pat_no_top_alt(expected).map_err(|mut err| {
                err.span_label(lo, WHILE_PARSING_OR_MSG);
                err
            })?;
            if rc == RecoverComma::Yes {
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
        let (pat, trailing_vert) = self.parse_pat_allow_top_alt_inner(
            expected,
            rc,
            RecoverColon::No,
            CommaRecoveryMode::LikelyTuple,
        )?;
        let colon = self.eat(&token::Colon);

        if let PatKind::Or(pats) = &pat.kind {
            let span = pat.span;

            if trailing_vert {
                // We already emitted an error and suggestion to remove the trailing vert. Don't
                // emit again.

                // FIXME(#100717): pass `TopLevelOrPatternNotAllowed::* { sub: None }` to
                // `delay_span_bug()` instead of fluent message
                self.sess.span_diagnostic.delay_span_bug(
                    span,
                    match syntax_loc {
                        PatternLocation::LetBinding => {
                            fluent::parse_or_pattern_not_allowed_in_let_binding
                        }
                        PatternLocation::FunctionParameter => {
                            fluent::parse_or_pattern_not_allowed_in_fn_parameters
                        }
                    },
                );
            } else {
                let pat = pprust::pat_to_string(&pat);
                let sub = if pats.len() == 1 {
                    Some(TopLevelOrPatternNotAllowedSugg::RemoveLeadingVert { span, pat })
                } else {
                    Some(TopLevelOrPatternNotAllowedSugg::WrapInParens { span, pat })
                };

                self.sess.emit_err(match syntax_loc {
                    PatternLocation::LetBinding => {
                        TopLevelOrPatternNotAllowed::LetBinding { span, sub }
                    }
                    PatternLocation::FunctionParameter => {
                        TopLevelOrPatternNotAllowed::FunctionParameter { span, sub }
                    }
                });
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
            self.sess.emit_err(UnexpectedVertVertBeforeFunctionParam { span: self.token.span });
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
        } else if matches!(self.token.kind, token::OrOr) {
            // Found `||`; Recover and pretend we parsed `|`.
            self.sess.emit_err(UnexpectedVertVertInPattern { span: self.token.span, start: lo });
            self.bump();
            EatOrResult::AteOr
        } else if self.eat(&token::BinOp(token::Or)) {
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
                | token::Ident(kw::If, false) // e.g. `a | if expr`.
                | token::Eq // e.g. `let a | = 0`.
                | token::Semi // e.g. `let a |;`.
                | token::Colon // e.g. `let a | :`.
                | token::Comma // e.g. `let (a |,)`.
                | token::CloseDelim(Delimiter::Bracket) // e.g. `let [a | ]`.
                | token::CloseDelim(Delimiter::Parenthesis) // e.g. `let (a | )`.
                | token::CloseDelim(Delimiter::Brace) // e.g. `let A { f: a | }`.
            )
        });
        match (is_end_ahead, &self.token.kind) {
            (true, token::BinOp(token::Or) | token::OrOr) => {
                // A `|` or possibly `||` token shouldn't be here. Ban it.
                self.sess.emit_err(TrailingVertNotAllowed {
                    span: self.token.span,
                    start: lo,
                    token: self.token.clone(),
                    note_double_vert: matches!(self.token.kind, token::OrOr).then_some(()),
                });
                self.bump();
                true
            }
            _ => false,
        }
    }

    /// Parses a pattern, with a setting whether modern range patterns (e.g., `a..=b`, `a..b` are
    /// allowed).
    fn parse_pat_with_range_pat(
        &mut self,
        allow_range_pat: bool,
        expected: Option<Expected>,
    ) -> PResult<'a, P<Pat>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);
        maybe_whole!(self, NtPat, |x| x);

        let mut lo = self.token.span;

        if self.token.is_keyword(kw::Let) && self.look_ahead(1, |tok| tok.can_begin_pattern()) {
            self.bump();
            self.sess.emit_err(RemoveLet { span: lo });
            lo = self.token.span;
        }

        let pat = if self.check(&token::BinOp(token::And)) || self.token.kind == token::AndAnd {
            self.parse_pat_deref(expected)?
        } else if self.check(&token::OpenDelim(Delimiter::Parenthesis)) {
            self.parse_pat_tuple_or_parens()?
        } else if self.check(&token::OpenDelim(Delimiter::Bracket)) {
            // Parse `[pat, pat,...]` as a slice pattern.
            let (pats, _) = self.parse_delim_comma_seq(Delimiter::Bracket, |p| {
                p.parse_pat_allow_top_alt(
                    None,
                    RecoverComma::No,
                    RecoverColon::No,
                    CommaRecoveryMode::EitherTupleOrPipe,
                )
            })?;
            PatKind::Slice(pats)
        } else if self.check(&token::DotDot) && !self.is_pat_range_end_start(1) {
            // A rest pattern `..`.
            self.bump(); // `..`
            PatKind::Rest
        } else if self.check(&token::DotDotDot) && !self.is_pat_range_end_start(1) {
            self.recover_dotdotdot_rest_pat(lo)
        } else if let Some(form) = self.parse_range_end() {
            self.parse_pat_range_to(form)? // `..=X`, `...X`, or `..X`.
        } else if self.eat_keyword(kw::Underscore) {
            // Parse _
            PatKind::Wild
        } else if self.eat_keyword(kw::Mut) {
            self.parse_pat_ident_mut()?
        } else if self.eat_keyword(kw::Ref) {
            // Parse ref ident @ pat / ref mut ident @ pat
            let mutbl = self.parse_mutability();
            self.parse_pat_ident(BindingAnnotation(ByRef::Yes, mutbl))?
        } else if self.eat_keyword(kw::Box) {
            self.parse_pat_box()?
        } else if self.check_inline_const(0) {
            // Parse `const pat`
            let const_expr = self.parse_const_block(lo.to(self.token.span), true)?;

            if let Some(re) = self.parse_range_end() {
                self.parse_pat_range_begin_with(const_expr, re)?
            } else {
                PatKind::Lit(const_expr)
            }
        // Don't eagerly error on semantically invalid tokens when matching
        // declarative macros, as the input to those doesn't have to be
        // semantically valid. For attribute/derive proc macros this is not the
        // case, so doing the recovery for them is fine.
        } else if self.can_be_ident_pat()
            || (self.is_lit_bad_ident().is_some() && self.may_recover())
        {
            // Parse `ident @ pat`
            // This can give false positives and parse nullary enums,
            // they are dealt with later in resolve.
            self.parse_pat_ident(BindingAnnotation::NONE)?
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

            if qself.is_none() && self.check(&token::Not) {
                self.parse_pat_mac_invoc(path)?
            } else if let Some(form) = self.parse_range_end() {
                let begin = self.mk_expr(span, ExprKind::Path(qself, path));
                self.parse_pat_range_begin_with(begin, form)?
            } else if self.check(&token::OpenDelim(Delimiter::Brace)) {
                self.parse_pat_struct(qself, path)?
            } else if self.check(&token::OpenDelim(Delimiter::Parenthesis)) {
                self.parse_pat_tuple_struct(qself, path)?
            } else {
                PatKind::Path(qself, path)
            }
        } else if matches!(self.token.kind, token::Lifetime(_))
            // In pattern position, we're totally fine with using "next token isn't colon"
            // as a heuristic. We could probably just always try to recover if it's a lifetime,
            // because we never have `'a: label {}` in a pattern position anyways, but it does
            // keep us from suggesting something like `let 'a: Ty = ..` => `let 'a': Ty = ..`
            && !self.look_ahead(1, |token| matches!(token.kind, token::Colon))
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

                    let mut err = self_.struct_span_err(self_.token.span, msg);
                    err.span_label(self_.token.span, format!("expected {}", expected));
                    err
                });
            PatKind::Lit(self.mk_expr(lo, ExprKind::Lit(lit)))
        } else {
            // Try to parse everything else as literal with optional minus
            match self.parse_literal_maybe_minus() {
                Ok(begin) => match self.parse_range_end() {
                    Some(form) => self.parse_pat_range_begin_with(begin, form)?,
                    None => PatKind::Lit(begin),
                },
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
        self.sess.emit_err(DotDotDotRestPattern { span: lo });
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
        if self.token.kind != token::At {
            // Next token is not `@` so it's not going to be an intersection pattern.
            return Ok(lhs);
        }

        // At this point we attempt to parse `@ $pat_rhs` and emit an error.
        self.bump(); // `@`
        let mut rhs = self.parse_pat_no_top_alt(None)?;
        let whole_span = lhs.span.to(rhs.span);

        if let PatKind::Ident(_, _, sub @ None) = &mut rhs.kind {
            // The user inverted the order, so help them fix that.
            let lhs_span = lhs.span;
            // Move the LHS into the RHS as a subpattern.
            // The RHS is now the full pattern.
            *sub = Some(lhs);

            self.sess.emit_err(PatternOnWrongSideOfAt {
                whole_span,
                whole_pat: pprust::pat_to_string(&rhs),
                pattern: lhs_span,
                binding: rhs.span,
            });
        } else {
            // The special case above doesn't apply so we may have e.g. `A(x) @ B(y)`.
            rhs.kind = PatKind::Wild;
            self.sess.emit_err(ExpectedBindingLeftOfAt {
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

        self.sess
            .emit_err(AmbiguousRangePattern { span: pat.span, pat: pprust::pat_to_string(&pat) });
    }

    /// Parse `&pat` / `&mut pat`.
    fn parse_pat_deref(&mut self, expected: Option<Expected>) -> PResult<'a, PatKind> {
        self.expect_and()?;
        if let token::Lifetime(name) = self.token.kind {
            self.bump(); // `'a`

            self.sess
                .emit_err(UnexpectedLifetimeInPattern { span: self.prev_token.span, symbol: name });
        }

        let mutbl = self.parse_mutability();
        let subpat = self.parse_pat_with_range_pat(false, expected)?;
        Ok(PatKind::Ref(subpat, mutbl))
    }

    /// Parse a tuple or parenthesis pattern.
    fn parse_pat_tuple_or_parens(&mut self) -> PResult<'a, PatKind> {
        let (fields, trailing_comma) = self.parse_paren_comma_seq(|p| {
            p.parse_pat_allow_top_alt(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::LikelyTuple,
            )
        })?;

        // Here, `(pat,)` is a tuple pattern.
        // For backward compatibility, `(..)` is a tuple pattern as well.
        Ok(if fields.len() == 1 && !(trailing_comma || fields[0].is_rest()) {
            PatKind::Paren(fields.into_iter().next().unwrap())
        } else {
            PatKind::Tuple(fields)
        })
    }

    /// Parse a mutable binding with the `mut` token already eaten.
    fn parse_pat_ident_mut(&mut self) -> PResult<'a, PatKind> {
        let mut_span = self.prev_token.span;

        if self.eat_keyword(kw::Ref) {
            self.sess.emit_err(RefMutOrderIncorrect { span: mut_span.to(self.prev_token.span) });
            return self.parse_pat_ident(BindingAnnotation::REF_MUT);
        }

        self.recover_additional_muts();

        // Make sure we don't allow e.g. `let mut $p;` where `$p:pat`.
        if let token::Interpolated(nt) = &self.token.kind {
            if let token::NtPat(_) = **nt {
                self.expected_ident_found_err().emit();
            }
        }

        // Parse the pattern we hope to be an identifier.
        let mut pat = self.parse_pat_no_top_alt(Some(Expected::Identifier))?;

        // If we don't have `mut $ident (@ pat)?`, error.
        if let PatKind::Ident(BindingAnnotation(ByRef::No, m @ Mutability::Not), ..) = &mut pat.kind
        {
            // Don't recurse into the subpattern.
            // `mut` on the outer binding doesn't affect the inner bindings.
            *m = Mutability::Mut;
        } else {
            // Add `mut` to any binding in the parsed pattern.
            let changed_any_binding = Self::make_all_value_bindings_mutable(&mut pat);
            self.ban_mut_general_pat(mut_span, &pat, changed_any_binding);
        }

        Ok(pat.into_inner().kind)
    }

    /// Turn all by-value immutable bindings in a pattern into mutable bindings.
    /// Returns `true` if any change was made.
    fn make_all_value_bindings_mutable(pat: &mut P<Pat>) -> bool {
        struct AddMut(bool);
        impl MutVisitor for AddMut {
            fn visit_pat(&mut self, pat: &mut P<Pat>) {
                if let PatKind::Ident(BindingAnnotation(ByRef::No, m @ Mutability::Not), ..) =
                    &mut pat.kind
                {
                    self.0 = true;
                    *m = Mutability::Mut;
                }
                noop_visit_pat(pat, self);
            }
        }

        let mut add_mut = AddMut(false);
        add_mut.visit_pat(pat);
        add_mut.0
    }

    /// Error on `mut $pat` where `$pat` is not an ident.
    fn ban_mut_general_pat(&self, lo: Span, pat: &Pat, changed_any_binding: bool) {
        let span = lo.to(pat.span);
        let pat = pprust::pat_to_string(&pat);

        self.sess.emit_err(if changed_any_binding {
            InvalidMutInPattern::NestedIdent { span, pat }
        } else {
            InvalidMutInPattern::NonIdent { span, pat }
        });
    }

    /// Eat any extraneous `mut`s and error + recover if we ate any.
    fn recover_additional_muts(&mut self) {
        let lo = self.token.span;
        while self.eat_keyword(kw::Mut) {}
        if lo == self.token.span {
            return;
        }

        self.sess.emit_err(RepeatedMutInPattern { span: lo.to(self.prev_token.span) });
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
        err: DiagnosticBuilder<'a, ErrorGuaranteed>,
        expected: Option<Expected>,
    ) -> PResult<'a, P<Pat>> {
        err.cancel();

        let expected = Expected::to_string_or_fallback(expected);
        let msg = format!("expected {}, found {}", expected, super::token_descr(&self.token));

        let mut err = self.struct_span_err(self.token.span, msg);
        err.span_label(self.token.span, format!("expected {}", expected));

        let sp = self.sess.source_map().start_point(self.token.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
        }

        Err(err)
    }

    /// Parses the range pattern end form `".." | "..." | "..=" ;`.
    fn parse_range_end(&mut self) -> Option<Spanned<RangeEnd>> {
        let re = if self.eat(&token::DotDotDot) {
            RangeEnd::Included(RangeSyntax::DotDotDot)
        } else if self.eat(&token::DotDotEq) {
            RangeEnd::Included(RangeSyntax::DotDotEq)
        } else if self.eat(&token::DotDot) {
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

    pub(super) fn inclusive_range_with_incorrect_end(&mut self) {
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

                self.sess.emit_err(InclusiveRangeExtraEquals { span: span_with_eq });
            }
            token::Gt if no_space => {
                let after_pat = span.with_hi(span.hi() - rustc_span::BytePos(1)).shrink_to_hi();
                self.sess.emit_err(InclusiveRangeMatchArrow { span, arrow: tok.span, after_pat });
            }
            _ => {
                self.sess.emit_err(InclusiveRangeNoEnd { span });
            }
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
            self.sess.emit_err(DotDotDotRangeToPatternNotAllowed { span: re.span });
        }
        Ok(PatKind::Range(None, Some(end), re))
    }

    /// Is the token `dist` away from the current suitable as the start of a range patterns end?
    fn is_pat_range_end_start(&self, dist: usize) -> bool {
        self.check_inline_const(dist)
            || self.look_ahead(dist, |t| {
                t.is_path_start() // e.g. `MY_CONST`;
                || t.kind == token::Dot // e.g. `.5` for recovery;
                || t.can_begin_literal_maybe_minus() // e.g. `42`.
                || t.is_whole_expr()
                || t.is_lifetime() // recover `'a` instead of `'a'`
            })
    }

    fn parse_pat_range_end(&mut self) -> PResult<'a, P<Expr>> {
        if self.check_inline_const(0) {
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
        }
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
        && self.look_ahead(1, |t| !matches!(t.kind, token::OpenDelim(Delimiter::Parenthesis) // A tuple struct pattern.
            | token::OpenDelim(Delimiter::Brace) // A struct pattern.
            | token::DotDotDot | token::DotDotEq | token::DotDot // A range pattern.
            | token::ModSep // A tuple / struct variant pattern.
            | token::Not)) // A macro expanding to a pattern.
    }

    /// Parses `ident` or `ident @ pat`.
    /// Used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like `ref foo(a, b)`.
    fn parse_pat_ident(&mut self, binding_annotation: BindingAnnotation) -> PResult<'a, PatKind> {
        let ident = self.parse_ident()?;
        let sub = if self.eat(&token::At) {
            Some(self.parse_pat_no_top_alt(Some(Expected::BindingPattern))?)
        } else {
            None
        };

        // Just to be friendly, if they write something like `ref Some(i)`,
        // we end up here with `(` as the current token.
        // This shortly leads to a parse error. Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to `parse_enum_variant()`.
        if self.token == token::OpenDelim(Delimiter::Parenthesis) {
            return Err(EnumPatternInsteadOfIdentifier { span: self.prev_token.span }
                .into_diagnostic(&self.sess.span_diagnostic));
        }

        Ok(PatKind::Ident(binding_annotation, ident, sub))
    }

    /// Parse a struct ("record") pattern (e.g. `Foo { ... }` or `Foo::Bar { ... }`).
    fn parse_pat_struct(&mut self, qself: Option<P<QSelf>>, path: Path) -> PResult<'a, PatKind> {
        if qself.is_some() {
            // Feature gate the use of qualified paths in patterns
            self.sess.gated_spans.gate(sym::more_qualified_paths, path.span);
        }
        self.bump();
        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
            e.span_label(path.span, "while parsing the fields for this pattern");
            e.emit();
            self.recover_stmt();
            (ThinVec::new(), true)
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
            p.parse_pat_allow_top_alt(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::EitherTupleOrPipe,
            )
        })?;
        if qself.is_some() {
            self.sess.gated_spans.gate(sym::more_qualified_paths, path.span);
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
            token::OpenDelim(Delimiter::Brace),
            token::CloseDelim(Delimiter::Brace),
            token::CloseDelim(Delimiter::Parenthesis),
        ]
        .contains(&self.token.kind)
    }

    /// Parses `box pat`
    fn parse_pat_box(&mut self) -> PResult<'a, PatKind> {
        let box_span = self.prev_token.span;

        if self.isnt_pattern_start() {
            let descr = super::token_descr(&self.token);
            self.sess.emit_err(errors::BoxNotPat {
                span: self.token.span,
                kw: box_span,
                lo: box_span.shrink_to_lo(),
                descr,
            });

            // We cannot use `parse_pat_ident()` since it will complain `box`
            // is not an identifier.
            let sub = if self.eat(&token::At) {
                Some(self.parse_pat_no_top_alt(Some(Expected::BindingPattern))?)
            } else {
                None
            };

            Ok(PatKind::Ident(BindingAnnotation::NONE, Ident::new(kw::Box, box_span), sub))
        } else {
            let pat = self.parse_pat_with_range_pat(false, None)?;
            self.sess.gated_spans.gate(sym::box_patterns, box_span.to(self.prev_token.span));
            Ok(PatKind::Box(pat))
        }
    }

    /// Parses the fields of a struct-like pattern.
    fn parse_pat_fields(&mut self) -> PResult<'a, (ThinVec<PatField>, bool)> {
        let mut fields = ThinVec::new();
        let mut etc = false;
        let mut ate_comma = true;
        let mut delayed_err: Option<DiagnosticBuilder<'a, ErrorGuaranteed>> = None;
        let mut first_etc_and_maybe_comma_span = None;
        let mut last_non_comma_dotdot_span = None;

        while self.token != token::CloseDelim(Delimiter::Brace) {
            let attrs = match self.parse_outer_attributes() {
                Ok(attrs) => attrs,
                Err(err) => {
                    if let Some(mut delayed) = delayed_err {
                        delayed.emit();
                    }
                    return Err(err);
                }
            };
            let lo = self.token.span;

            // check that a comma comes after every field
            if !ate_comma {
                let err = ExpectedCommaAfterPatternField { span: self.token.span }
                    .into_diagnostic(&self.sess.span_diagnostic);
                if let Some(mut delayed) = delayed_err {
                    delayed.emit();
                }
                return Err(err);
            }
            ate_comma = false;

            if self.check(&token::DotDot)
                || self.check_noexpect(&token::DotDotDot)
                || self.check_keyword(kw::Underscore)
            {
                etc = true;
                let mut etc_sp = self.token.span;
                if first_etc_and_maybe_comma_span.is_none() {
                    if let Some(comma_tok) = self
                        .look_ahead(1, |t| if *t == token::Comma { Some(t.clone()) } else { None })
                    {
                        let nw_span = self
                            .sess
                            .source_map()
                            .span_extend_to_line(comma_tok.span)
                            .trim_start(comma_tok.span.shrink_to_lo())
                            .map(|s| self.sess.source_map().span_until_non_whitespace(s));
                        first_etc_and_maybe_comma_span = nw_span.map(|s| etc_sp.to(s));
                    } else {
                        first_etc_and_maybe_comma_span =
                            Some(self.sess.source_map().span_until_non_whitespace(etc_sp));
                    }
                }

                self.recover_bad_dot_dot();
                self.bump(); // `..` || `...` || `_`

                if self.token == token::CloseDelim(Delimiter::Brace) {
                    break;
                }
                let token_str = super::token_descr(&self.token);
                let msg = format!("expected `}}`, found {}", token_str);
                let mut err = self.struct_span_err(self.token.span, msg);

                err.span_label(self.token.span, "expected `}`");
                let mut comma_sp = None;
                if self.token == token::Comma {
                    // Issue #49257
                    let nw_span = self.sess.source_map().span_until_non_whitespace(self.token.span);
                    etc_sp = etc_sp.to(nw_span);
                    err.span_label(
                        etc_sp,
                        "`..` must be at the end and cannot have a trailing comma",
                    );
                    comma_sp = Some(self.token.span);
                    self.bump();
                    ate_comma = true;
                }

                if self.token == token::CloseDelim(Delimiter::Brace) {
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
                    if let Some(mut delayed_err) = delayed_err {
                        delayed_err.emit();
                        return Err(err);
                    } else {
                        delayed_err = Some(err);
                    }
                } else {
                    if let Some(mut err) = delayed_err {
                        err.emit();
                    }
                    return Err(err);
                }
            }

            let field =
                self.collect_tokens_trailing_token(attrs, ForceCollect::No, |this, attrs| {
                    let field = match this.parse_pat_field(lo, attrs) {
                        Ok(field) => Ok(field),
                        Err(err) => {
                            if let Some(mut delayed_err) = delayed_err.take() {
                                delayed_err.emit();
                            }
                            return Err(err);
                        }
                    }?;
                    ate_comma = this.eat(&token::Comma);

                    last_non_comma_dotdot_span = Some(this.prev_token.span);

                    // We just ate a comma, so there's no need to use
                    // `TrailingToken::Comma`
                    Ok((field, TrailingToken::None))
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
                } else {
                    if let Some(last_non_comma_dotdot_span) = last_non_comma_dotdot_span {
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
            }
            err.emit();
        }
        Ok((fields, etc))
    }

    /// Recover on `...` or `_` as if it were `..` to avoid further errors.
    /// See issue #46718.
    fn recover_bad_dot_dot(&self) {
        if self.token == token::DotDot {
            return;
        }

        let token_str = pprust::token_to_string(&self.token);
        self.sess.emit_err(DotDotDotForRemainingFields { span: self.token.span, token_str });
    }

    fn parse_pat_field(&mut self, lo: Span, attrs: AttrVec) -> PResult<'a, PatField> {
        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let hi;
        let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
            // Parsing a pattern of the form `fieldname: pat`.
            let fieldname = self.parse_field_name()?;
            self.bump();
            let pat = self.parse_pat_allow_top_alt(
                None,
                RecoverComma::No,
                RecoverColon::No,
                CommaRecoveryMode::EitherTupleOrPipe,
            )?;
            hi = pat.span;
            (pat, fieldname, false)
        } else {
            // Parsing a pattern of the form `(box) (ref) (mut) fieldname`.
            let is_box = self.eat_keyword(kw::Box);
            let boxed_span = self.token.span;
            let is_ref = self.eat_keyword(kw::Ref);
            let is_mut = self.eat_keyword(kw::Mut);
            let fieldname = self.parse_field_name()?;
            hi = self.prev_token.span;

            let mutability = match is_mut {
                false => Mutability::Not,
                true => Mutability::Mut,
            };
            let ann = BindingAnnotation(ByRef::from(is_ref), mutability);
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

    pub(super) fn mk_pat_ident(&self, span: Span, ann: BindingAnnotation, ident: Ident) -> P<Pat> {
        self.mk_pat(span, PatKind::Ident(ann, ident, None))
    }

    pub(super) fn mk_pat(&self, span: Span, kind: PatKind) -> P<Pat> {
        P(Pat { kind, span, id: ast::DUMMY_NODE_ID, tokens: None })
    }
}
