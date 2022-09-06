use super::{ForceCollect, Parser, PathStyle, TrailingToken};
use crate::{maybe_recover_from_interpolated_ty_qpath, maybe_whole};
use rustc_ast::mut_visit::{noop_visit_pat, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter};
use rustc_ast::{
    self as ast, AttrVec, BindingAnnotation, ByRef, Expr, ExprKind, MacCall, Mutability, Pat,
    PatField, PatKind, Path, QSelf, RangeEnd, RangeSyntax,
};
use rustc_ast_pretty::pprust;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, ErrorGuaranteed, PResult};
use rustc_span::source_map::{respan, Span, Spanned};
use rustc_span::symbol::{kw, sym, Ident};

pub(super) type Expected = Option<&'static str>;

/// `Expected` for function and lambda parameter patterns.
pub(super) const PARAM_EXPECTED: Expected = Some("parameter name");

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

impl<'a> Parser<'a> {
    /// Parses a pattern.
    ///
    /// Corresponds to `pat<no_top_alt>` in RFC 2535 and does not admit or-patterns
    /// at the top level. Used when parsing the parameters of lambda expressions,
    /// functions, function pointers, and `pat` macro fragments.
    pub fn parse_pat_no_top_alt(&mut self, expected: Expected) -> PResult<'a, P<Pat>> {
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
        expected: Expected,
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
        expected: Expected,
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
                return Ok((self.mk_pat(span, PatKind::Or(vec![first_pat])), trailing_vert));
            }

            return Ok((first_pat, trailing_vert));
        }

        // Parse the patterns `p_1 | ... | p_n` where `n > 0`.
        let lo = leading_vert_span.unwrap_or(first_pat.span);
        let mut pats = vec![first_pat];
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
        expected: Expected,
        rc: RecoverComma,
        syntax_loc: &str,
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
            let msg = format!("top-level or-patterns are not allowed in {}", syntax_loc);
            let (help, fix) = if pats.len() == 1 {
                // If all we have is a leading vert, then print a special message. This is the case
                // if `parse_pat_allow_top_alt` returns an or-pattern with one variant.
                let msg = "remove the `|`";
                let fix = pprust::pat_to_string(&pat);
                (msg, fix)
            } else {
                let msg = "wrap the pattern in parentheses";
                let fix = format!("({})", pprust::pat_to_string(&pat));
                (msg, fix)
            };

            if trailing_vert {
                // We already emitted an error and suggestion to remove the trailing vert. Don't
                // emit again.
                self.sess.span_diagnostic.delay_span_bug(pat.span, &msg);
            } else {
                self.struct_span_err(pat.span, &msg)
                    .span_suggestion(pat.span, help, fix, Applicability::MachineApplicable)
                    .emit();
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
            let span = self.token.span;
            let mut err = self.struct_span_err(span, "unexpected `||` before function parameter");
            err.span_suggestion(span, "remove the `||`", "", Applicability::MachineApplicable);
            err.note("alternatives in or-patterns are separated with `|`, not `||`");
            err.emit();
            self.bump();
        }

        self.parse_pat_before_ty(PARAM_EXPECTED, RecoverComma::No, "function parameters")
    }

    /// Eat the or-pattern `|` separator.
    /// If instead a `||` token is encountered, recover and pretend we parsed `|`.
    fn eat_or_separator(&mut self, lo: Option<Span>) -> EatOrResult {
        if self.recover_trailing_vert(lo) {
            EatOrResult::TrailingVert
        } else if matches!(self.token.kind, token::OrOr) {
            // Found `||`; Recover and pretend we parsed `|`.
            self.ban_unexpected_or_or(lo);
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
                self.ban_illegal_vert(lo, "trailing", "not allowed in an or-pattern");
                self.bump();
                true
            }
            _ => false,
        }
    }

    /// We have parsed `||` instead of `|`. Error and suggest `|` instead.
    fn ban_unexpected_or_or(&mut self, lo: Option<Span>) {
        let mut err = self.struct_span_err(self.token.span, "unexpected token `||` in pattern");
        err.span_suggestion(
            self.token.span,
            "use a single `|` to separate multiple alternative patterns",
            "|",
            Applicability::MachineApplicable,
        );
        if let Some(lo) = lo {
            err.span_label(lo, WHILE_PARSING_OR_MSG);
        }
        err.emit();
    }

    /// A `|` or possibly `||` token shouldn't be here. Ban it.
    fn ban_illegal_vert(&mut self, lo: Option<Span>, pos: &str, ctx: &str) {
        let span = self.token.span;
        let mut err = self.struct_span_err(span, &format!("a {} `|` is {}", pos, ctx));
        err.span_suggestion(
            span,
            &format!("remove the `{}`", pprust::token_to_string(&self.token)),
            "",
            Applicability::MachineApplicable,
        );
        if let Some(lo) = lo {
            err.span_label(lo, WHILE_PARSING_OR_MSG);
        }
        if let token::OrOr = self.token.kind {
            err.note("alternatives in or-patterns are separated with `|`, not `||`");
        }
        err.emit();
    }

    /// Parses a pattern, with a setting whether modern range patterns (e.g., `a..=b`, `a..b` are
    /// allowed).
    fn parse_pat_with_range_pat(
        &mut self,
        allow_range_pat: bool,
        expected: Expected,
    ) -> PResult<'a, P<Pat>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);
        maybe_whole!(self, NtPat, |x| x);

        let lo = self.token.span;

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
        } else if self.can_be_ident_pat() {
            // Parse `ident @ pat`
            // This can give false positives and parse nullary enums,
            // they are dealt with later in resolve.
            self.parse_pat_ident(BindingAnnotation::NONE)?
        } else if self.is_start_of_pat_with_path() {
            // Parse pattern starting with a path
            let (qself, path) = if self.eat_lt() {
                // Parse a qualified path
                let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, self.parse_path(PathStyle::Expr)?)
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
        self.struct_span_err(lo, "unexpected `...`")
            .span_label(lo, "not a valid pattern")
            .span_suggestion_short(
                lo,
                "for a rest pattern, use `..` instead of `...`",
                "..",
                Applicability::MachineApplicable,
            )
            .emit();
        PatKind::Rest
    }

    /// Try to recover the more general form `intersect ::= $pat_lhs @ $pat_rhs`.
    ///
    /// Allowed binding patterns generated by `binding ::= ref? mut? $ident @ $pat_rhs`
    /// should already have been parsed by now  at this point,
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
        let sp = lhs.span.to(rhs.span);

        if let PatKind::Ident(_, _, ref mut sub @ None) = rhs.kind {
            // The user inverted the order, so help them fix that.
            let mut applicability = Applicability::MachineApplicable;
            // FIXME(bindings_after_at): Remove this code when stabilizing the feature.
            lhs.walk(&mut |p| match p.kind {
                // `check_match` is unhappy if the subpattern has a binding anywhere.
                PatKind::Ident(..) => {
                    applicability = Applicability::MaybeIncorrect;
                    false // Short-circuit.
                }
                _ => true,
            });

            let lhs_span = lhs.span;
            // Move the LHS into the RHS as a subpattern.
            // The RHS is now the full pattern.
            *sub = Some(lhs);

            self.struct_span_err(sp, "pattern on wrong side of `@`")
                .span_label(lhs_span, "pattern on the left, should be on the right")
                .span_label(rhs.span, "binding on the right, should be on the left")
                .span_suggestion(sp, "switch the order", pprust::pat_to_string(&rhs), applicability)
                .emit();
        } else {
            // The special case above doesn't apply so we may have e.g. `A(x) @ B(y)`.
            rhs.kind = PatKind::Wild;
            self.struct_span_err(sp, "left-hand side of `@` must be a binding")
                .span_label(lhs.span, "interpreted as a pattern, not a binding")
                .span_label(rhs.span, "also a pattern")
                .note("bindings are `x`, `mut x`, `ref x`, and `ref mut x`")
                .emit();
        }

        rhs.span = sp;
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

        self.struct_span_err(pat.span, "the range pattern here has ambiguous interpretation")
            .span_suggestion(
                pat.span,
                "add parentheses to clarify the precedence",
                format!("({})", pprust::pat_to_string(&pat)),
                // "ambiguous interpretation" implies that we have to be guessing
                Applicability::MaybeIncorrect,
            )
            .emit();
    }

    /// Parse `&pat` / `&mut pat`.
    fn parse_pat_deref(&mut self, expected: Expected) -> PResult<'a, PatKind> {
        self.expect_and()?;
        self.recover_lifetime_in_deref_pat();
        let mutbl = self.parse_mutability();
        let subpat = self.parse_pat_with_range_pat(false, expected)?;
        Ok(PatKind::Ref(subpat, mutbl))
    }

    fn recover_lifetime_in_deref_pat(&mut self) {
        if let token::Lifetime(name) = self.token.kind {
            self.bump(); // `'a`

            let span = self.prev_token.span;
            self.struct_span_err(span, &format!("unexpected lifetime `{}` in pattern", name))
                .span_suggestion(span, "remove the lifetime", "", Applicability::MachineApplicable)
                .emit();
        }
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
            return self.recover_mut_ref_ident(mut_span);
        }

        self.recover_additional_muts();

        // Make sure we don't allow e.g. `let mut $p;` where `$p:pat`.
        if let token::Interpolated(ref nt) = self.token.kind {
            if let token::NtPat(_) = **nt {
                self.expected_ident_found().emit();
            }
        }

        // Parse the pattern we hope to be an identifier.
        let mut pat = self.parse_pat_no_top_alt(Some("identifier"))?;

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

    /// Recover on `mut ref? ident @ pat` and suggest
    /// that the order of `mut` and `ref` is incorrect.
    fn recover_mut_ref_ident(&mut self, lo: Span) -> PResult<'a, PatKind> {
        let mutref_span = lo.to(self.prev_token.span);
        self.struct_span_err(mutref_span, "the order of `mut` and `ref` is incorrect")
            .span_suggestion(
                mutref_span,
                "try switching the order",
                "ref mut",
                Applicability::MachineApplicable,
            )
            .emit();

        self.parse_pat_ident(BindingAnnotation::REF_MUT)
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
        let fix = pprust::pat_to_string(&pat);
        let (problem, suggestion) = if changed_any_binding {
            ("`mut` must be attached to each individual binding", "add `mut` to each binding")
        } else {
            ("`mut` must be followed by a named binding", "remove the `mut` prefix")
        };
        self.struct_span_err(span, problem)
            .span_suggestion(span, suggestion, fix, Applicability::MachineApplicable)
            .note("`mut` may be followed by `variable` and `variable @ pattern`")
            .emit();
    }

    /// Eat any extraneous `mut`s and error + recover if we ate any.
    fn recover_additional_muts(&mut self) {
        let lo = self.token.span;
        while self.eat_keyword(kw::Mut) {}
        if lo == self.token.span {
            return;
        }

        let span = lo.to(self.prev_token.span);
        self.struct_span_err(span, "`mut` on a binding may not be repeated")
            .span_suggestion(
                span,
                "remove the additional `mut`s",
                "",
                Applicability::MachineApplicable,
            )
            .emit();
    }

    /// Parse macro invocation
    fn parse_pat_mac_invoc(&mut self, path: Path) -> PResult<'a, PatKind> {
        self.bump();
        let args = self.parse_mac_args()?;
        let mac = P(MacCall { path, args, prior_type_ascription: self.last_type_ascription });
        Ok(PatKind::MacCall(mac))
    }

    fn fatal_unexpected_non_pat(
        &mut self,
        err: DiagnosticBuilder<'a, ErrorGuaranteed>,
        expected: Expected,
    ) -> PResult<'a, P<Pat>> {
        err.cancel();

        let expected = expected.unwrap_or("pattern");
        let msg = format!("expected {}, found {}", expected, super::token_descr(&self.token));

        let mut err = self.struct_span_err(self.token.span, &msg);
        err.span_label(self.token.span, format!("expected {}", expected));

        let sp = self.sess.source_map().start_point(self.token.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            self.sess.expr_parentheses_needed(&mut err, *sp);
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
                self.inclusive_range_with_incorrect_end(re.span);
            }
            None
        };
        Ok(PatKind::Range(Some(begin), end, re))
    }

    pub(super) fn inclusive_range_with_incorrect_end(&mut self, span: Span) {
        let tok = &self.token;

        // If the user typed "..==" instead of "..=", we want to give them
        // a specific error message telling them to use "..=".
        // Otherwise, we assume that they meant to type a half open exclusive
        // range and give them an error telling them to do that instead.
        if matches!(tok.kind, token::Eq) && tok.span.lo() == span.hi() {
            let span_with_eq = span.to(tok.span);

            // Ensure the user doesn't receive unhelpful unexpected token errors
            self.bump();
            if self.is_pat_range_end_start(0) {
                let _ = self.parse_pat_range_end().map_err(|e| e.cancel());
            }

            self.error_inclusive_range_with_extra_equals(span_with_eq);
        } else {
            self.error_inclusive_range_with_no_end(span);
        }
    }

    fn error_inclusive_range_with_extra_equals(&self, span: Span) {
        self.struct_span_err(span, "unexpected `=` after inclusive range")
            .span_suggestion_short(span, "use `..=` instead", "..=", Applicability::MaybeIncorrect)
            .note("inclusive ranges end with a single equals sign (`..=`)")
            .emit();
    }

    fn error_inclusive_range_with_no_end(&self, span: Span) {
        struct_span_err!(self.sess.span_diagnostic, span, E0586, "inclusive range with no end")
            .span_suggestion_short(span, "use `..` instead", "..", Applicability::MachineApplicable)
            .note("inclusive ranges must be bounded at the end (`..=b` or `a..=b`)")
            .emit();
    }

    /// Parse a range-to pattern, `..X` or `..=X` where `X` remains to be parsed.
    ///
    /// The form `...X` is prohibited to reduce confusion with the potential
    /// expression syntax `...expr` for splatting in expressions.
    fn parse_pat_range_to(&mut self, mut re: Spanned<RangeEnd>) -> PResult<'a, PatKind> {
        let end = self.parse_pat_range_end()?;
        self.sess.gated_spans.gate(sym::half_open_range_patterns, re.span.to(self.prev_token.span));
        if let RangeEnd::Included(ref mut syn @ RangeSyntax::DotDotDot) = &mut re.node {
            *syn = RangeSyntax::DotDotEq;
            self.struct_span_err(re.span, "range-to patterns with `...` are not allowed")
                .span_suggestion_short(
                    re.span,
                    "use `..=` instead",
                    "..=",
                    Applicability::MachineApplicable,
                )
                .emit();
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
            })
    }

    fn parse_pat_range_end(&mut self) -> PResult<'a, P<Expr>> {
        if self.check_inline_const(0) {
            self.parse_const_block(self.token.span, true)
        } else if self.check_path() {
            let lo = self.token.span;
            let (qself, path) = if self.eat_lt() {
                // Parse a qualified path
                let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, self.parse_path(PathStyle::Expr)?)
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
            Some(self.parse_pat_no_top_alt(Some("binding pattern"))?)
        } else {
            None
        };

        // Just to be friendly, if they write something like `ref Some(i)`,
        // we end up here with `(` as the current token.
        // This shortly leads to a parse error. Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to `parse_enum_variant()`.
        if self.token == token::OpenDelim(Delimiter::Parenthesis) {
            return Err(self
                .struct_span_err(self.prev_token.span, "expected identifier, found enum pattern"));
        }

        Ok(PatKind::Ident(binding_annotation, ident, sub))
    }

    /// Parse a struct ("record") pattern (e.g. `Foo { ... }` or `Foo::Bar { ... }`).
    fn parse_pat_struct(&mut self, qself: Option<QSelf>, path: Path) -> PResult<'a, PatKind> {
        if qself.is_some() {
            // Feature gate the use of qualified paths in patterns
            self.sess.gated_spans.gate(sym::more_qualified_paths, path.span);
        }
        self.bump();
        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
            e.span_label(path.span, "while parsing the fields for this pattern");
            e.emit();
            self.recover_stmt();
            (vec![], true)
        });
        self.bump();
        Ok(PatKind::Struct(qself, path, fields, etc))
    }

    /// Parse tuple struct or tuple variant pattern (e.g. `Foo(...)` or `Foo::Bar(...)`).
    fn parse_pat_tuple_struct(&mut self, qself: Option<QSelf>, path: Path) -> PResult<'a, PatKind> {
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
            self.struct_span_err(
                self.token.span,
                format!("expected pattern, found {}", super::token_descr(&self.token)),
            )
            .span_note(box_span, "`box` is a reserved keyword")
            .span_suggestion_verbose(
                box_span.shrink_to_lo(),
                "escape `box` to use it as an identifier",
                "r#",
                Applicability::MaybeIncorrect,
            )
            .emit();

            // We cannot use `parse_pat_ident()` since it will complain `box`
            // is not an identifier.
            let sub = if self.eat(&token::At) {
                Some(self.parse_pat_no_top_alt(Some("binding pattern"))?)
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
    fn parse_pat_fields(&mut self) -> PResult<'a, (Vec<PatField>, bool)> {
        let mut fields = Vec::new();
        let mut etc = false;
        let mut ate_comma = true;
        let mut delayed_err: Option<DiagnosticBuilder<'a, ErrorGuaranteed>> = None;
        let mut etc_span = None;

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
                let err = self.struct_span_err(self.token.span, "expected `,`");
                if let Some(mut delayed) = delayed_err {
                    delayed.emit();
                }
                return Err(err);
            }
            ate_comma = false;

            if self.check(&token::DotDot) || self.token == token::DotDotDot {
                etc = true;
                let mut etc_sp = self.token.span;

                self.recover_one_fewer_dotdot();
                self.bump(); // `..` || `...`

                if self.token == token::CloseDelim(Delimiter::Brace) {
                    etc_span = Some(etc_sp);
                    break;
                }
                let token_str = super::token_descr(&self.token);
                let msg = &format!("expected `}}`, found {}", token_str);
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

                etc_span = Some(etc_sp.until(self.token.span));
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
                    // We just ate a comma, so there's no need to use
                    // `TrailingToken::Comma`
                    Ok((field, TrailingToken::None))
                })?;

            fields.push(field)
        }

        if let Some(mut err) = delayed_err {
            if let Some(etc_span) = etc_span {
                err.multipart_suggestion(
                    "move the `..` to the end of the field list",
                    vec![
                        (etc_span, String::new()),
                        (self.token.span, format!("{}.. }}", if ate_comma { "" } else { ", " })),
                    ],
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        }
        Ok((fields, etc))
    }

    /// Recover on `...` as if it were `..` to avoid further errors.
    /// See issue #46718.
    fn recover_one_fewer_dotdot(&self) {
        if self.token != token::DotDotDot {
            return;
        }

        self.struct_span_err(self.token.span, "expected field pattern, found `...`")
            .span_suggestion(
                self.token.span,
                "to omit remaining fields, use one fewer `.`",
                "..",
                Applicability::MachineApplicable,
            )
            .emit();
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
