use super::{Parser, PathStyle};
use crate::{maybe_recover_from_interpolated_ty_qpath, maybe_whole};
use rustc_ast::mut_visit::{noop_visit_pat, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::{self as ast, AttrVec, Attribute, FieldPat, MacCall, Pat, PatKind, RangeEnd};
use rustc_ast::{BindingMode, Expr, ExprKind, Mutability, Path, QSelf, RangeSyntax};
use rustc_ast_pretty::pprust;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, PResult};
use rustc_span::source_map::{respan, Span, Spanned};
use rustc_span::symbol::{kw, sym, Ident};

type Expected = Option<&'static str>;

/// `Expected` for function and lambda parameter patterns.
pub(super) const PARAM_EXPECTED: Expected = Some("parameter name");

const WHILE_PARSING_OR_MSG: &str = "while parsing this or-pattern starting here";

/// Whether or not an or-pattern should be gated when occurring in the current context.
#[derive(PartialEq, Clone, Copy)]
pub(super) enum GateOr {
    Yes,
    No,
}

/// Whether or not to recover a `,` when parsing or-patterns.
#[derive(PartialEq, Copy, Clone)]
pub(super) enum RecoverComma {
    Yes,
    No,
}

impl<'a> Parser<'a> {
    /// Parses a pattern.
    ///
    /// Corresponds to `pat<no_top_alt>` in RFC 2535 and does not admit or-patterns
    /// at the top level. Used when parsing the parameters of lambda expressions,
    /// functions, function pointers, and `pat` macro fragments.
    pub fn parse_pat(&mut self, expected: Expected) -> PResult<'a, P<Pat>> {
        self.parse_pat_with_range_pat(true, expected)
    }

    /// Entry point to the main pattern parser.
    /// Corresponds to `top_pat` in RFC 2535 and allows or-pattern at the top level.
    pub(super) fn parse_top_pat(
        &mut self,
        gate_or: GateOr,
        rc: RecoverComma,
    ) -> PResult<'a, P<Pat>> {
        // Allow a '|' before the pats (RFCs 1925, 2530, and 2535).
        let gated_leading_vert = self.eat_or_separator(None) && gate_or == GateOr::Yes;
        let leading_vert_span = self.prev_token.span;

        // Parse the possibly-or-pattern.
        let pat = self.parse_pat_with_or(None, gate_or, rc)?;

        // If we parsed a leading `|` which should be gated,
        // and no other gated or-pattern has been parsed thus far,
        // then we should really gate the leading `|`.
        // This complicated procedure is done purely for diagnostics UX.
        if gated_leading_vert && self.sess.gated_spans.is_ungated(sym::or_patterns) {
            self.sess.gated_spans.gate(sym::or_patterns, leading_vert_span);
        }

        Ok(pat)
    }

    /// Parse the pattern for a function or function pointer parameter.
    /// Special recovery is provided for or-patterns and leading `|`.
    pub(super) fn parse_fn_param_pat(&mut self) -> PResult<'a, P<Pat>> {
        self.recover_leading_vert(None, "not allowed in a parameter pattern");
        let pat = self.parse_pat_with_or(PARAM_EXPECTED, GateOr::No, RecoverComma::No)?;

        if let PatKind::Or(..) = &pat.kind {
            self.ban_illegal_fn_param_or_pat(&pat);
        }

        Ok(pat)
    }

    /// Ban `A | B` immediately in a parameter pattern and suggest wrapping in parens.
    fn ban_illegal_fn_param_or_pat(&self, pat: &Pat) {
        let msg = "wrap the pattern in parenthesis";
        let fix = format!("({})", pprust::pat_to_string(pat));
        self.struct_span_err(pat.span, "an or-pattern parameter must be wrapped in parenthesis")
            .span_suggestion(pat.span, msg, fix, Applicability::MachineApplicable)
            .emit();
    }

    /// Parses a pattern, that may be a or-pattern (e.g. `Foo | Bar` in `Some(Foo | Bar)`).
    /// Corresponds to `pat<allow_top_alt>` in RFC 2535.
    fn parse_pat_with_or(
        &mut self,
        expected: Expected,
        gate_or: GateOr,
        rc: RecoverComma,
    ) -> PResult<'a, P<Pat>> {
        // Parse the first pattern (`p_0`).
        let first_pat = self.parse_pat(expected)?;
        self.maybe_recover_unexpected_comma(first_pat.span, rc, gate_or)?;

        // If the next token is not a `|`,
        // this is not an or-pattern and we should exit here.
        if !self.check(&token::BinOp(token::Or)) && self.token != token::OrOr {
            return Ok(first_pat);
        }

        // Parse the patterns `p_1 | ... | p_n` where `n > 0`.
        let lo = first_pat.span;
        let mut pats = vec![first_pat];
        while self.eat_or_separator(Some(lo)) {
            let pat = self.parse_pat(expected).map_err(|mut err| {
                err.span_label(lo, WHILE_PARSING_OR_MSG);
                err
            })?;
            self.maybe_recover_unexpected_comma(pat.span, rc, gate_or)?;
            pats.push(pat);
        }
        let or_pattern_span = lo.to(self.prev_token.span);

        // Feature gate the or-pattern if instructed:
        if gate_or == GateOr::Yes {
            self.sess.gated_spans.gate(sym::or_patterns, or_pattern_span);
        }

        Ok(self.mk_pat(or_pattern_span, PatKind::Or(pats)))
    }

    /// Eat the or-pattern `|` separator.
    /// If instead a `||` token is encountered, recover and pretend we parsed `|`.
    fn eat_or_separator(&mut self, lo: Option<Span>) -> bool {
        if self.recover_trailing_vert(lo) {
            return false;
        }

        match self.token.kind {
            token::OrOr => {
                // Found `||`; Recover and pretend we parsed `|`.
                self.ban_unexpected_or_or(lo);
                self.bump();
                true
            }
            _ => self.eat(&token::BinOp(token::Or)),
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
            | token::CloseDelim(token::Bracket) // e.g. `let [a | ]`.
            | token::CloseDelim(token::Paren) // e.g. `let (a | )`.
            | token::CloseDelim(token::Brace) // e.g. `let A { f: a | }`.
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
        let mut err = self.struct_span_err(self.token.span, "unexpected token `||` after pattern");
        err.span_suggestion(
            self.token.span,
            "use a single `|` to separate multiple alternative patterns",
            "|".to_owned(),
            Applicability::MachineApplicable,
        );
        if let Some(lo) = lo {
            err.span_label(lo, WHILE_PARSING_OR_MSG);
        }
        err.emit();
    }

    /// Some special error handling for the "top-level" patterns in a match arm,
    /// `for` loop, `let`, &c. (in contrast to subpatterns within such).
    fn maybe_recover_unexpected_comma(
        &mut self,
        lo: Span,
        rc: RecoverComma,
        gate_or: GateOr,
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

            let or_suggestion =
                gate_or == GateOr::No || !self.sess.gated_spans.is_ungated(sym::or_patterns);
            err.span_suggestion(
                seq_span,
                if or_suggestion { MSG } else { MSG.trim_end_matches('.') },
                format!("({})", seq_snippet),
                Applicability::MachineApplicable,
            );
            if or_suggestion {
                err.span_suggestion(
                    seq_span,
                    "...or a vertical bar to match on multiple alternatives",
                    seq_snippet.replace(",", " |"),
                    Applicability::MachineApplicable,
                );
            }
        }
        Err(err)
    }

    /// Parse and throw away a parentesized comma separated
    /// sequence of patterns until `)` is reached.
    fn skip_pat_list(&mut self) -> PResult<'a, ()> {
        while !self.check(&token::CloseDelim(token::Paren)) {
            self.parse_pat(None)?;
            if !self.eat(&token::Comma) {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Recursive possibly-or-pattern parser with recovery for an erroneous leading `|`.
    /// See `parse_pat_with_or` for details on parsing or-patterns.
    fn parse_pat_with_or_inner(&mut self) -> PResult<'a, P<Pat>> {
        self.recover_leading_vert(None, "only allowed in a top-level pattern");
        self.parse_pat_with_or(None, GateOr::Yes, RecoverComma::No)
    }

    /// Recover if `|` or `||` is here.
    /// The user is thinking that a leading `|` is allowed in this position.
    fn recover_leading_vert(&mut self, lo: Option<Span>, ctx: &str) {
        if let token::BinOp(token::Or) | token::OrOr = self.token.kind {
            self.ban_illegal_vert(lo, "leading", ctx);
            self.bump();
        }
    }

    /// A `|` or possibly `||` token shouldn't be here. Ban it.
    fn ban_illegal_vert(&mut self, lo: Option<Span>, pos: &str, ctx: &str) {
        let span = self.token.span;
        let mut err = self.struct_span_err(span, &format!("a {} `|` is {}", pos, ctx));
        err.span_suggestion(
            span,
            &format!("remove the `{}`", pprust::token_to_string(&self.token)),
            String::new(),
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
        } else if self.check(&token::OpenDelim(token::Paren)) {
            self.parse_pat_tuple_or_parens()?
        } else if self.check(&token::OpenDelim(token::Bracket)) {
            // Parse `[pat, pat,...]` as a slice pattern.
            let (pats, _) =
                self.parse_delim_comma_seq(token::Bracket, |p| p.parse_pat_with_or_inner())?;
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
            self.parse_pat_ident(BindingMode::ByRef(mutbl))?
        } else if self.eat_keyword(kw::Box) {
            // Parse `box pat`
            let pat = self.parse_pat_with_range_pat(false, None)?;
            self.sess.gated_spans.gate(sym::box_patterns, lo.to(self.prev_token.span));
            PatKind::Box(pat)
        } else if self.check_inline_const(0) {
            // Parse `const pat`
            let const_expr = self.parse_const_block(lo.to(self.token.span))?;

            if let Some(re) = self.parse_range_end() {
                self.parse_pat_range_begin_with(const_expr, re)?
            } else {
                PatKind::Lit(const_expr)
            }
        } else if self.can_be_ident_pat() {
            // Parse `ident @ pat`
            // This can give false positives and parse nullary enums,
            // they are dealt with later in resolve.
            self.parse_pat_ident(BindingMode::ByValue(Mutability::Not))?
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
                let begin = self.mk_expr(span, ExprKind::Path(qself, path), AttrVec::new());
                self.parse_pat_range_begin_with(begin, form)?
            } else if self.check(&token::OpenDelim(token::Brace)) {
                self.parse_pat_struct(qself, path)?
            } else if self.check(&token::OpenDelim(token::Paren)) {
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
        let pat = self.maybe_recover_from_bad_qpath(pat, true)?;
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
                "..".to_owned(),
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
        let mut rhs = self.parse_pat(None)?;
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
                .span_suggestion(
                    span,
                    "remove the lifetime",
                    String::new(),
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }

    /// Parse a tuple or parenthesis pattern.
    fn parse_pat_tuple_or_parens(&mut self) -> PResult<'a, PatKind> {
        let (fields, trailing_comma) =
            self.parse_paren_comma_seq(|p| p.parse_pat_with_or_inner())?;

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
        let mut pat = self.parse_pat(Some("identifier"))?;

        // If we don't have `mut $ident (@ pat)?`, error.
        if let PatKind::Ident(BindingMode::ByValue(m @ Mutability::Not), ..) = &mut pat.kind {
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
                "ref mut".into(),
                Applicability::MachineApplicable,
            )
            .emit();

        self.parse_pat_ident(BindingMode::ByRef(Mutability::Mut))
    }

    /// Turn all by-value immutable bindings in a pattern into mutable bindings.
    /// Returns `true` if any change was made.
    fn make_all_value_bindings_mutable(pat: &mut P<Pat>) -> bool {
        struct AddMut(bool);
        impl MutVisitor for AddMut {
            fn visit_pat(&mut self, pat: &mut P<Pat>) {
                if let PatKind::Ident(BindingMode::ByValue(m @ Mutability::Not), ..) = &mut pat.kind
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
                String::new(),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    /// Parse macro invocation
    fn parse_pat_mac_invoc(&mut self, path: Path) -> PResult<'a, PatKind> {
        self.bump();
        let args = self.parse_mac_args()?;
        let mac = MacCall { path, args, prior_type_ascription: self.last_type_ascription };
        Ok(PatKind::MacCall(mac))
    }

    fn fatal_unexpected_non_pat(
        &mut self,
        mut err: DiagnosticBuilder<'a>,
        expected: Expected,
    ) -> PResult<'a, P<Pat>> {
        err.cancel();

        let expected = expected.unwrap_or("pattern");
        let msg = format!("expected {}, found {}", expected, super::token_descr(&self.token));

        let mut err = self.struct_span_err(self.token.span, &msg);
        err.span_label(self.token.span, format!("expected {}", expected));

        let sp = self.sess.source_map().start_point(self.token.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            self.sess.expr_parentheses_needed(&mut err, *sp, None);
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
            self.sess.gated_spans.gate(sym::exclusive_range_pattern, self.prev_token.span);
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
            self.sess.gated_spans.gate(sym::half_open_range_patterns, begin.span.to(re.span));
            if let RangeEnd::Included(_) = re.node {
                // FIXME(Centril): Consider semantic errors instead in `ast_validation`.
                // Possibly also do this for `X..=` in *expression* contexts.
                self.error_inclusive_range_with_no_end(re.span);
            }
            None
        };
        Ok(PatKind::Range(Some(begin), end, re))
    }

    pub(super) fn error_inclusive_range_with_no_end(&self, span: Span) {
        struct_span_err!(self.sess.span_diagnostic, span, E0586, "inclusive range with no end")
            .span_suggestion_short(
                span,
                "use `..` instead",
                "..".to_string(),
                Applicability::MachineApplicable,
            )
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
                    "..=".to_string(),
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
            self.parse_const_block(self.token.span)
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
            Ok(self.mk_expr(lo.to(hi), ExprKind::Path(qself, path), AttrVec::new()))
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
        && self.look_ahead(1, |t| !matches!(t.kind, token::OpenDelim(token::Paren) // A tuple struct pattern.
            | token::OpenDelim(token::Brace) // A struct pattern.
            | token::DotDotDot | token::DotDotEq | token::DotDot // A range pattern.
            | token::ModSep // A tuple / struct variant pattern.
            | token::Not)) // A macro expanding to a pattern.
    }

    /// Parses `ident` or `ident @ pat`.
    /// Used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like `ref foo(a, b)`.
    fn parse_pat_ident(&mut self, binding_mode: BindingMode) -> PResult<'a, PatKind> {
        let ident = self.parse_ident()?;
        let sub = if self.eat(&token::At) {
            Some(self.parse_pat(Some("binding pattern"))?)
        } else {
            None
        };

        // Just to be friendly, if they write something like `ref Some(i)`,
        // we end up here with `(` as the current token.
        // This shortly leads to a parse error. Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to `parse_enum_variant()`.
        if self.token == token::OpenDelim(token::Paren) {
            return Err(self
                .struct_span_err(self.prev_token.span, "expected identifier, found enum pattern"));
        }

        Ok(PatKind::Ident(binding_mode, ident, sub))
    }

    /// Parse a struct ("record") pattern (e.g. `Foo { ... }` or `Foo::Bar { ... }`).
    fn parse_pat_struct(&mut self, qself: Option<QSelf>, path: Path) -> PResult<'a, PatKind> {
        if qself.is_some() {
            return self.error_qpath_before_pat(&path, "{");
        }
        self.bump();
        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
            e.span_label(path.span, "while parsing the fields for this pattern");
            e.emit();
            self.recover_stmt();
            (vec![], true)
        });
        self.bump();
        Ok(PatKind::Struct(path, fields, etc))
    }

    /// Parse tuple struct or tuple variant pattern (e.g. `Foo(...)` or `Foo::Bar(...)`).
    fn parse_pat_tuple_struct(&mut self, qself: Option<QSelf>, path: Path) -> PResult<'a, PatKind> {
        if qself.is_some() {
            return self.error_qpath_before_pat(&path, "(");
        }
        let (fields, _) = self.parse_paren_comma_seq(|p| p.parse_pat_with_or_inner())?;
        Ok(PatKind::TupleStruct(path, fields))
    }

    /// Error when there's a qualified path, e.g. `<Foo as Bar>::Baz`
    /// as the path of e.g., a tuple or record struct pattern.
    fn error_qpath_before_pat(&mut self, path: &Path, token: &str) -> PResult<'a, PatKind> {
        let msg = &format!("unexpected `{}` after qualified path", token);
        let mut err = self.struct_span_err(self.token.span, msg);
        err.span_label(self.token.span, msg);
        err.span_label(path.span, "the qualified path");
        Err(err)
    }

    /// Parses the fields of a struct-like pattern.
    fn parse_pat_fields(&mut self) -> PResult<'a, (Vec<FieldPat>, bool)> {
        let mut fields = Vec::new();
        let mut etc = false;
        let mut ate_comma = true;
        let mut delayed_err: Option<DiagnosticBuilder<'a>> = None;
        let mut etc_span = None;

        while self.token != token::CloseDelim(token::Brace) {
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

                if self.token == token::CloseDelim(token::Brace) {
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
                if self.token == token::CloseDelim(token::Brace) {
                    // If the struct looks otherwise well formed, recover and continue.
                    if let Some(sp) = comma_sp {
                        err.span_suggestion_short(
                            sp,
                            "remove this comma",
                            String::new(),
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

            fields.push(match self.parse_pat_field(lo, attrs) {
                Ok(field) => field,
                Err(err) => {
                    if let Some(mut delayed_err) = delayed_err {
                        delayed_err.emit();
                    }
                    return Err(err);
                }
            });
            ate_comma = self.eat(&token::Comma);
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
                "..".to_owned(),
                Applicability::MachineApplicable,
            )
            .emit();
    }

    fn parse_pat_field(&mut self, lo: Span, attrs: Vec<Attribute>) -> PResult<'a, FieldPat> {
        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let hi;
        let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
            // Parsing a pattern of the form `fieldname: pat`.
            let fieldname = self.parse_field_name()?;
            self.bump();
            let pat = self.parse_pat_with_or_inner()?;
            hi = pat.span;
            (pat, fieldname, false)
        } else {
            // Parsing a pattern of the form `(box) (ref) (mut) fieldname`.
            let is_box = self.eat_keyword(kw::Box);
            let boxed_span = self.token.span;
            let is_ref = self.eat_keyword(kw::Ref);
            let is_mut = self.eat_keyword(kw::Mut);
            let fieldname = self.parse_ident()?;
            hi = self.prev_token.span;

            let bind_type = match (is_ref, is_mut) {
                (true, true) => BindingMode::ByRef(Mutability::Mut),
                (true, false) => BindingMode::ByRef(Mutability::Not),
                (false, true) => BindingMode::ByValue(Mutability::Mut),
                (false, false) => BindingMode::ByValue(Mutability::Not),
            };

            let fieldpat = self.mk_pat_ident(boxed_span.to(hi), bind_type, fieldname);
            let subpat =
                if is_box { self.mk_pat(lo.to(hi), PatKind::Box(fieldpat)) } else { fieldpat };
            (subpat, fieldname, true)
        };

        Ok(FieldPat {
            ident: fieldname,
            pat: subpat,
            is_shorthand,
            attrs: attrs.into(),
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            is_placeholder: false,
        })
    }

    pub(super) fn mk_pat_ident(&self, span: Span, bm: BindingMode, ident: Ident) -> P<Pat> {
        self.mk_pat(span, PatKind::Ident(bm, ident, None))
    }

    fn mk_pat(&self, span: Span, kind: PatKind) -> P<Pat> {
        P(Pat { kind, span, id: ast::DUMMY_NODE_ID, tokens: None })
    }
}
