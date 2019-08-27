use super::{Parser, PResult, PathStyle};

use crate::{maybe_recover_from_interpolated_ty_qpath, maybe_whole};
use crate::ptr::P;
use crate::ast::{self, Attribute, Pat, PatKind, FieldPat, RangeEnd, RangeSyntax, Mac};
use crate::ast::{BindingMode, Ident, Mutability, Path, QSelf, Expr, ExprKind};
use crate::mut_visit::{noop_visit_pat, MutVisitor};
use crate::parse::token::{self};
use crate::print::pprust;
use crate::source_map::{respan, Span, Spanned};
use crate::symbol::kw;
use crate::ThinVec;

use errors::{Applicability, DiagnosticBuilder};

type Expected = Option<&'static str>;

/// `Expected` for function and lambda parameter patterns.
pub(super) const PARAM_EXPECTED: Expected = Some("parameter name");

/// Whether or not an or-pattern should be gated when occurring in the current context.
#[derive(PartialEq)]
pub enum GateOr { Yes, No }

/// Whether or not to recover a `,` when parsing or-patterns.
#[derive(PartialEq, Copy, Clone)]
enum RecoverComma { Yes, No }

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
    pub(super) fn parse_top_pat(&mut self, gate_or: GateOr) -> PResult<'a, P<Pat>> {
        // Allow a '|' before the pats (RFCs 1925, 2530, and 2535).
        let gated_leading_vert = self.eat_or_separator() && gate_or == GateOr::Yes;
        let leading_vert_span = self.prev_span;

        // Parse the possibly-or-pattern.
        let pat = self.parse_pat_with_or(None, gate_or, RecoverComma::Yes)?;

        // If we parsed a leading `|` which should be gated,
        // and no other gated or-pattern has been parsed thus far,
        // then we should really gate the leading `|`.
        // This complicated procedure is done purely for diagnostics UX.
        if gated_leading_vert {
            let mut or_pattern_spans = self.sess.gated_spans.or_patterns.borrow_mut();
            if or_pattern_spans.is_empty() {
                or_pattern_spans.push(leading_vert_span);
            }
        }

        Ok(pat)
    }

    /// Parse the pattern for a function or function pointer parameter.
    /// Special recovery is provided for or-patterns and leading `|`.
    pub(super) fn parse_fn_param_pat(&mut self) -> PResult<'a, P<Pat>> {
        self.recover_leading_vert("not allowed in a parameter pattern");
        let pat = self.parse_pat_with_or(PARAM_EXPECTED, GateOr::No, RecoverComma::No)?;

        if let PatKind::Or(..) = &pat.node {
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
        // Parse the first pattern.
        let first_pat = self.parse_pat(expected)?;
        self.maybe_recover_unexpected_comma(first_pat.span, rc)?;

        // If the next token is not a `|`,
        // this is not an or-pattern and we should exit here.
        if !self.check(&token::BinOp(token::Or)) && self.token != token::OrOr {
            return Ok(first_pat)
        }

        let lo = first_pat.span;
        let mut pats = vec![first_pat];
        while self.eat_or_separator() {
            let pat = self.parse_pat(expected).map_err(|mut err| {
                err.span_label(lo, "while parsing this or-pattern starting here");
                err
            })?;
            self.maybe_recover_unexpected_comma(pat.span, rc)?;
            pats.push(pat);
        }
        let or_pattern_span = lo.to(self.prev_span);

        // Feature gate the or-pattern if instructed:
        if gate_or == GateOr::Yes {
            self.sess.gated_spans.or_patterns.borrow_mut().push(or_pattern_span);
        }

        Ok(self.mk_pat(or_pattern_span, PatKind::Or(pats)))
    }

    /// Eat the or-pattern `|` separator.
    /// If instead a `||` token is encountered, recover and pretend we parsed `|`.
    fn eat_or_separator(&mut self) -> bool {
        match self.token.kind {
            token::OrOr => {
                // Found `||`; Recover and pretend we parsed `|`.
                self.ban_unexpected_or_or();
                self.bump();
                true
            }
            _ => self.eat(&token::BinOp(token::Or)),
        }
    }

    /// We have parsed `||` instead of `|`. Error and suggest `|` instead.
    fn ban_unexpected_or_or(&mut self) {
        self.struct_span_err(self.token.span, "unexpected token `||` after pattern")
            .span_suggestion(
                self.token.span,
                "use a single `|` to separate multiple alternative patterns",
                "|".to_owned(),
                Applicability::MachineApplicable
            )
            .emit();
    }

    /// Some special error handling for the "top-level" patterns in a match arm,
    /// `for` loop, `let`, &c. (in contrast to subpatterns within such).
    fn maybe_recover_unexpected_comma(&mut self, lo: Span, rc: RecoverComma) -> PResult<'a, ()> {
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
        let seq_span = lo.to(self.prev_span);
        let mut err = self.struct_span_err(comma_span, "unexpected `,` in pattern");
        if let Ok(seq_snippet) = self.span_to_snippet(seq_span) {
            err.span_suggestion(
                seq_span,
                "try adding parentheses to match on a tuple..",
                format!("({})", seq_snippet),
                Applicability::MachineApplicable
            )
            .span_suggestion(
                seq_span,
                "..or a vertical bar to match on multiple alternatives",
                format!("{}", seq_snippet.replace(",", " |")),
                Applicability::MachineApplicable
            );
        }
        Err(err)
    }

    /// Parse and throw away a parentesized comma separated
    /// sequence of patterns until `)` is reached.
    fn skip_pat_list(&mut self) -> PResult<'a, ()> {
        while !self.check(&token::CloseDelim(token::Paren)) {
            self.parse_pat(None)?;
            if !self.eat(&token::Comma) {
                return Ok(())
            }
        }
        Ok(())
    }

    /// Recursive possibly-or-pattern parser with recovery for an erroneous leading `|`.
    /// See `parse_pat_with_or` for details on parsing or-patterns.
    fn parse_pat_with_or_inner(&mut self) -> PResult<'a, P<Pat>> {
        self.recover_leading_vert("only allowed in a top-level pattern");
        self.parse_pat_with_or(None, GateOr::Yes, RecoverComma::No)
    }

    /// Recover if `|` or `||` is here.
    /// The user is thinking that a leading `|` is allowed in this position.
    fn recover_leading_vert(&mut self, ctx: &str) {
        if let token::BinOp(token::Or) | token::OrOr = self.token.kind {
            let span = self.token.span;
            let rm_msg = format!("remove the `{}`", pprust::token_to_string(&self.token));

            self.struct_span_err(span, &format!("a leading `|` is {}", ctx))
                .span_suggestion(span, &rm_msg, String::new(), Applicability::MachineApplicable)
                .emit();

            self.bump();
        }
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
        let pat = match self.token.kind {
            token::BinOp(token::And) | token::AndAnd => self.parse_pat_deref(expected)?,
            token::OpenDelim(token::Paren) => self.parse_pat_tuple_or_parens()?,
            token::OpenDelim(token::Bracket) => {
                // Parse `[pat, pat,...]` as a slice pattern.
                let (pats, _) = self.parse_delim_comma_seq(
                    token::Bracket,
                    |p| p.parse_pat_with_or_inner(),
                )?;
                PatKind::Slice(pats)
            }
            token::DotDot => {
                self.bump();
                if self.is_pat_range_end_start() {
                    // Parse `..42` for recovery.
                    self.parse_pat_range_to(RangeEnd::Excluded, "..")?
                } else {
                    // A rest pattern `..`.
                    PatKind::Rest
                }
            }
            token::DotDotEq => {
                // Parse `..=42` for recovery.
                self.bump();
                self.parse_pat_range_to(RangeEnd::Included(RangeSyntax::DotDotEq), "..=")?
            }
            token::DotDotDot => {
                // Parse `...42` for recovery.
                self.bump();
                self.parse_pat_range_to(RangeEnd::Included(RangeSyntax::DotDotDot), "...")?
            }
            // At this point, token != &, &&, (, [
            _ => if self.eat_keyword(kw::Underscore) {
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
                PatKind::Box(self.parse_pat_with_range_pat(false, None)?)
            } else if self.can_be_ident_pat() {
                // Parse `ident @ pat`
                // This can give false positives and parse nullary enums,
                // they are dealt with later in resolve.
                self.parse_pat_ident(BindingMode::ByValue(Mutability::Immutable))?
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
                match self.token.kind {
                    token::Not if qself.is_none() => self.parse_pat_mac_invoc(lo, path)?,
                    token::DotDotDot | token::DotDotEq | token::DotDot => {
                        self.parse_pat_range_starting_with_path(lo, qself, path)?
                    }
                    token::OpenDelim(token::Brace) => self.parse_pat_struct(qself, path)?,
                    token::OpenDelim(token::Paren) => self.parse_pat_tuple_struct(qself, path)?,
                    _ => PatKind::Path(qself, path),
                }
            } else {
                // Try to parse everything else as literal with optional minus
                match self.parse_literal_maybe_minus() {
                    Ok(begin)
                        if self.check(&token::DotDot)
                            || self.check(&token::DotDotEq)
                            || self.check(&token::DotDotDot) =>
                    {
                        self.parse_pat_range_starting_with_lit(begin)?
                    }
                    Ok(begin) => PatKind::Lit(begin),
                    Err(err) => return self.fatal_unexpected_non_pat(err, expected),
                }
            }
        };

        let pat = self.mk_pat(lo.to(self.prev_span), pat);
        let pat = self.maybe_recover_from_bad_qpath(pat, true)?;

        if !allow_range_pat {
            self.ban_pat_range_if_ambiguous(&pat)?
        }

        Ok(pat)
    }

    /// Ban a range pattern if it has an ambiguous interpretation.
    fn ban_pat_range_if_ambiguous(&self, pat: &Pat) -> PResult<'a, ()> {
        match pat.node {
            PatKind::Range(
                .., Spanned { node: RangeEnd::Included(RangeSyntax::DotDotDot), .. }
            ) => return Ok(()),
            PatKind::Range(..) => {}
            _ => return Ok(()),
        }

        let mut err = self.struct_span_err(
            pat.span,
            "the range pattern here has ambiguous interpretation",
        );
        err.span_suggestion(
            pat.span,
            "add parentheses to clarify the precedence",
            format!("({})", pprust::pat_to_string(&pat)),
            // "ambiguous interpretation" implies that we have to be guessing
            Applicability::MaybeIncorrect
        );
        Err(err)
    }

    /// Parse `&pat` / `&mut pat`.
    fn parse_pat_deref(&mut self, expected: Expected) -> PResult<'a, PatKind> {
        self.expect_and()?;
        let mutbl = self.parse_mutability();

        if let token::Lifetime(name) = self.token.kind {
            let mut err = self.fatal(&format!("unexpected lifetime `{}` in pattern", name));
            err.span_label(self.token.span, "unexpected lifetime");
            return Err(err);
        }

        let subpat = self.parse_pat_with_range_pat(false, expected)?;
        Ok(PatKind::Ref(subpat, mutbl))
    }

    /// Parse a tuple or parenthesis pattern.
    fn parse_pat_tuple_or_parens(&mut self) -> PResult<'a, PatKind> {
        let (fields, trailing_comma) = self.parse_paren_comma_seq(|p| p.parse_pat_with_or_inner())?;

        // Here, `(pat,)` is a tuple pattern.
        // For backward compatibility, `(..)` is a tuple pattern as well.
        Ok(if fields.len() == 1 && !(trailing_comma || fields[0].is_rest()) {
            PatKind::Paren(fields.into_iter().nth(0).unwrap())
        } else {
            PatKind::Tuple(fields)
        })
    }

    /// Parse a mutable binding with the `mut` token already eaten.
    fn parse_pat_ident_mut(&mut self) -> PResult<'a, PatKind> {
        let mut_span = self.prev_span;

        if self.eat_keyword(kw::Ref) {
            return self.recover_mut_ref_ident(mut_span)
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

        // Add `mut` to any binding in the parsed pattern.
        let changed_any_binding = Self::make_all_value_bindings_mutable(&mut pat);

        // Unwrap; If we don't have `mut $ident`, error.
        let pat = pat.into_inner();
        match &pat.node {
            PatKind::Ident(..) => {}
            _ => self.ban_mut_general_pat(mut_span, &pat, changed_any_binding),
        }

        Ok(pat.node)
    }

    /// Recover on `mut ref? ident @ pat` and suggest
    /// that the order of `mut` and `ref` is incorrect.
    fn recover_mut_ref_ident(&mut self, lo: Span) -> PResult<'a, PatKind> {
        let mutref_span = lo.to(self.prev_span);
        self.struct_span_err(mutref_span, "the order of `mut` and `ref` is incorrect")
            .span_suggestion(
                mutref_span,
                "try switching the order",
                "ref mut".into(),
                Applicability::MachineApplicable
            )
            .emit();

        self.parse_pat_ident(BindingMode::ByRef(Mutability::Mutable))
    }

    /// Turn all by-value immutable bindings in a pattern into mutable bindings.
    /// Returns `true` if any change was made.
    fn make_all_value_bindings_mutable(pat: &mut P<Pat>) -> bool {
        struct AddMut(bool);
        impl MutVisitor for AddMut {
            fn visit_pat(&mut self, pat: &mut P<Pat>) {
                if let PatKind::Ident(BindingMode::ByValue(ref mut m @ Mutability::Immutable), ..)
                    = pat.node
                {
                    *m = Mutability::Mutable;
                    self.0 = true;
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
            .emit()
    }

    /// Eat any extraneous `mut`s and error + recover if we ate any.
    fn recover_additional_muts(&mut self) {
        let lo = self.token.span;
        while self.eat_keyword(kw::Mut) {}
        if lo == self.token.span {
            return;
        }

        let span = lo.to(self.prev_span);
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
    fn parse_pat_mac_invoc(&mut self, lo: Span, path: Path) -> PResult<'a, PatKind> {
        self.bump();
        let (delim, tts) = self.expect_delimited_token_tree()?;
        let mac = Mac {
            path,
            tts,
            delim,
            span: lo.to(self.prev_span),
            prior_type_ascription: self.last_type_ascription,
        };
        Ok(PatKind::Mac(mac))
    }

    /// Parse a range pattern `$path $form $end?` where `$form = ".." | "..." | "..=" ;`.
    /// The `$path` has already been parsed and the next token is the `$form`.
    fn parse_pat_range_starting_with_path(
        &mut self,
        lo: Span,
        qself: Option<QSelf>,
        path: Path
    ) -> PResult<'a, PatKind> {
        let (end_kind, form) = match self.token.kind {
            token::DotDot => (RangeEnd::Excluded, ".."),
            token::DotDotDot => (RangeEnd::Included(RangeSyntax::DotDotDot), "..."),
            token::DotDotEq => (RangeEnd::Included(RangeSyntax::DotDotEq), "..="),
            _ => panic!("can only parse `..`/`...`/`..=` for ranges (checked above)"),
        };
        let op_span = self.token.span;
        // Parse range
        let span = lo.to(self.prev_span);
        let begin = self.mk_expr(span, ExprKind::Path(qself, path), ThinVec::new());
        self.bump();
        let end = self.parse_pat_range_end_opt(&begin, form)?;
        Ok(PatKind::Range(begin, end, respan(op_span, end_kind)))
    }

    /// Parse a range pattern `$literal $form $end?` where `$form = ".." | "..." | "..=" ;`.
    /// The `$path` has already been parsed and the next token is the `$form`.
    fn parse_pat_range_starting_with_lit(&mut self, begin: P<Expr>) -> PResult<'a, PatKind> {
        let op_span = self.token.span;
        let (end_kind, form) = if self.eat(&token::DotDotDot) {
            (RangeEnd::Included(RangeSyntax::DotDotDot), "...")
        } else if self.eat(&token::DotDotEq) {
            (RangeEnd::Included(RangeSyntax::DotDotEq), "..=")
        } else if self.eat(&token::DotDot) {
            (RangeEnd::Excluded, "..")
        } else {
            panic!("impossible case: we already matched on a range-operator token")
        };
        let end = self.parse_pat_range_end_opt(&begin, form)?;
        Ok(PatKind::Range(begin, end, respan(op_span, end_kind)))
    }

    fn fatal_unexpected_non_pat(
        &mut self,
        mut err: DiagnosticBuilder<'a>,
        expected: Expected,
    ) -> PResult<'a, P<Pat>> {
        self.cancel(&mut err);

        let expected = expected.unwrap_or("pattern");
        let msg = format!("expected {}, found {}", expected, self.this_token_descr());

        let mut err = self.fatal(&msg);
        err.span_label(self.token.span, format!("expected {}", expected));

        let sp = self.sess.source_map().start_point(self.token.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            self.sess.expr_parentheses_needed(&mut err, *sp, None);
        }

        Err(err)
    }

    /// Is the current token suitable as the start of a range patterns end?
    fn is_pat_range_end_start(&self) -> bool {
        self.token.is_path_start() // e.g. `MY_CONST`;
            || self.token == token::Dot // e.g. `.5` for recovery;
            || self.token.can_begin_literal_or_bool() // e.g. `42`.
            || self.token.is_whole_expr()
    }

    /// Parse a range-to pattern, e.g. `..X` and `..=X` for recovery.
    fn parse_pat_range_to(&mut self, re: RangeEnd, form: &str) -> PResult<'a, PatKind> {
        let lo = self.prev_span;
        let end = self.parse_pat_range_end()?;
        let range_span = lo.to(end.span);
        let begin = self.mk_expr(range_span, ExprKind::Err, ThinVec::new());

        self.diagnostic()
            .struct_span_err(range_span, &format!("`{}X` range patterns are not supported", form))
            .span_suggestion(
                range_span,
                "try using the minimum value for the type",
                format!("MIN{}{}", form, pprust::expr_to_string(&end)),
                Applicability::HasPlaceholders,
            )
            .emit();

        Ok(PatKind::Range(begin, end, respan(lo, re)))
    }

    /// Parse the end of a `X..Y`, `X..=Y`, or `X...Y` range pattern  or recover
    /// if that end is missing treating it as `X..`, `X..=`, or `X...` respectively.
    fn parse_pat_range_end_opt(&mut self, begin: &Expr, form: &str) -> PResult<'a, P<Expr>> {
        if self.is_pat_range_end_start() {
            // Parsing e.g. `X..=Y`.
            self.parse_pat_range_end()
        } else {
            // Parsing e.g. `X..`.
            let range_span = begin.span.to(self.prev_span);

            self.diagnostic()
                .struct_span_err(
                    range_span,
                    &format!("`X{}` range patterns are not supported", form),
                )
                .span_suggestion(
                    range_span,
                    "try using the maximum value for the type",
                    format!("{}{}MAX", pprust::expr_to_string(&begin), form),
                    Applicability::HasPlaceholders,
                )
                .emit();

            Ok(self.mk_expr(range_span, ExprKind::Err, ThinVec::new()))
        }
    }

    fn parse_pat_range_end(&mut self) -> PResult<'a, P<Expr>> {
        if self.token.is_path_start() {
            let lo = self.token.span;
            let (qself, path) = if self.eat_lt() {
                // Parse a qualified path
                let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, self.parse_path(PathStyle::Expr)?)
            };
            let hi = self.prev_span;
            Ok(self.mk_expr(lo.to(hi), ExprKind::Path(qself, path), ThinVec::new()))
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
        && self.look_ahead(1, |t| match t.kind { // Try to do something more complex?
            token::OpenDelim(token::Paren) // A tuple struct pattern.
            | token::OpenDelim(token::Brace) // A struct pattern.
            | token::DotDotDot | token::DotDotEq | token::DotDot // A range pattern.
            | token::ModSep // A tuple / struct variant pattern.
            | token::Not => false, // A macro expanding to a pattern.
            _ => true,
        })
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
            return Err(self.span_fatal(
                self.prev_span,
                "expected identifier, found enum pattern",
            ))
        }

        Ok(PatKind::Ident(binding_mode, ident, sub))
    }

    /// Parse a struct ("record") pattern (e.g. `Foo { ... }` or `Foo::Bar { ... }`).
    fn parse_pat_struct(&mut self, qself: Option<QSelf>, path: Path) -> PResult<'a, PatKind> {
        if qself.is_some() {
            let msg = "unexpected `{` after qualified path";
            let mut err = self.fatal(msg);
            err.span_label(self.token.span, msg);
            return Err(err);
        }

        self.bump();
        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
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
            let msg = "unexpected `(` after qualified path";
            let mut err = self.fatal(msg);
            err.span_label(self.token.span, msg);
            return Err(err);
        }
        let (fields, _) = self.parse_paren_comma_seq(|p| p.parse_pat_with_or_inner())?;
        Ok(PatKind::TupleStruct(path, fields))
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
                },
            };
            let lo = self.token.span;

            // check that a comma comes after every field
            if !ate_comma {
                let err = self.struct_span_err(self.prev_span, "expected `,`");
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
                self.bump();  // `..` || `...`

                if self.token == token::CloseDelim(token::Brace) {
                    etc_span = Some(etc_sp);
                    break;
                }
                let token_str = self.this_token_descr();
                let mut err = self.fatal(&format!("expected `}}`, found {}", token_str));

                err.span_label(self.token.span, "expected `}`");
                let mut comma_sp = None;
                if self.token == token::Comma { // Issue #49257
                    let nw_span = self.sess.source_map().span_until_non_whitespace(self.token.span);
                    etc_sp = etc_sp.to(nw_span);
                    err.span_label(etc_sp,
                                   "`..` must be at the end and cannot have a trailing comma");
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
        return Ok((fields, etc));
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
                Applicability::MachineApplicable
            )
            .emit();
    }

    fn parse_pat_field(&mut self, lo: Span, attrs: Vec<Attribute>) -> PResult<'a, FieldPat> {
        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let hi;
        let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
            // Parsing a pattern of the form "fieldname: pat"
            let fieldname = self.parse_field_name()?;
            self.bump();
            let pat = self.parse_pat_with_or_inner()?;
            hi = pat.span;
            (pat, fieldname, false)
        } else {
            // Parsing a pattern of the form "(box) (ref) (mut) fieldname"
            let is_box = self.eat_keyword(kw::Box);
            let boxed_span = self.token.span;
            let is_ref = self.eat_keyword(kw::Ref);
            let is_mut = self.eat_keyword(kw::Mut);
            let fieldname = self.parse_ident()?;
            hi = self.prev_span;

            let bind_type = match (is_ref, is_mut) {
                (true, true) => BindingMode::ByRef(Mutability::Mutable),
                (true, false) => BindingMode::ByRef(Mutability::Immutable),
                (false, true) => BindingMode::ByValue(Mutability::Mutable),
                (false, false) => BindingMode::ByValue(Mutability::Immutable),
            };

            let fieldpat = self.mk_pat_ident(boxed_span.to(hi), bind_type, fieldname);
            let subpat = if is_box {
                self.mk_pat(lo.to(hi), PatKind::Box(fieldpat))
            } else {
                fieldpat
            };
            (subpat, fieldname, true)
        };

        Ok(FieldPat {
            ident: fieldname,
            pat: subpat,
            is_shorthand,
            attrs: attrs.into(),
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
        })
    }

    pub(super) fn mk_pat_ident(&self, span: Span, bm: BindingMode, ident: Ident) -> P<Pat> {
        self.mk_pat(span, PatKind::Ident(bm, ident, None))
    }

    fn mk_pat(&self, span: Span, node: PatKind) -> P<Pat> {
        P(Pat { node, span, id: ast::DUMMY_NODE_ID })
    }
}
