use super::{Parser, PResult, PathStyle};

use crate::{maybe_recover_from_interpolated_ty_qpath, maybe_whole};
use crate::ptr::P;
use crate::ast::{self, Attribute, Pat, PatKind, FieldPat, RangeEnd, RangeSyntax, Mac};
use crate::ast::{BindingMode, Ident, Mutability, Path, QSelf, Expr, ExprKind};
use crate::parse::token::{self};
use crate::print::pprust;
use crate::source_map::{respan, Span, Spanned};
use crate::symbol::kw;
use crate::ThinVec;

use errors::{Applicability, DiagnosticBuilder};

impl<'a> Parser<'a> {
    /// Parses a pattern.
    pub fn parse_pat(
        &mut self,
        expected: Option<&'static str>
    ) -> PResult<'a, P<Pat>> {
        self.parse_pat_with_range_pat(true, expected)
    }

    /// Parses patterns, separated by '|' s.
    pub(super) fn parse_pats(&mut self) -> PResult<'a, Vec<P<Pat>>> {
        // Allow a '|' before the pats (RFC 1925 + RFC 2530)
        self.eat(&token::BinOp(token::Or));

        let mut pats = Vec::new();
        loop {
            pats.push(self.parse_top_level_pat()?);

            if self.token == token::OrOr {
                self.struct_span_err(self.token.span, "unexpected token `||` after pattern")
                    .span_suggestion(
                        self.token.span,
                        "use a single `|` to specify multiple patterns",
                        "|".to_owned(),
                        Applicability::MachineApplicable
                    )
                    .emit();
                self.bump();
            } else if self.eat(&token::BinOp(token::Or)) {
                // This is a No-op. Continue the loop to parse the next
                // pattern.
            } else {
                return Ok(pats);
            }
        };
    }

    /// A wrapper around `parse_pat` with some special error handling for the
    /// "top-level" patterns in a match arm, `for` loop, `let`, &c. (in contrast
    /// to subpatterns within such).
    pub(super) fn parse_top_level_pat(&mut self) -> PResult<'a, P<Pat>> {
        let pat = self.parse_pat(None)?;
        if self.token == token::Comma {
            // An unexpected comma after a top-level pattern is a clue that the
            // user (perhaps more accustomed to some other language) forgot the
            // parentheses in what should have been a tuple pattern; return a
            // suggestion-enhanced error here rather than choking on the comma
            // later.
            let comma_span = self.token.span;
            self.bump();
            if let Err(mut err) = self.skip_pat_list() {
                // We didn't expect this to work anyway; we just wanted
                // to advance to the end of the comma-sequence so we know
                // the span to suggest parenthesizing
                err.cancel();
            }
            let seq_span = pat.span.to(self.prev_span);
            let mut err = self.struct_span_err(comma_span, "unexpected `,` in pattern");
            if let Ok(seq_snippet) = self.span_to_snippet(seq_span) {
                err.span_suggestion(
                    seq_span,
                    "try adding parentheses to match on a tuple..",
                    format!("({})", seq_snippet),
                    Applicability::MachineApplicable
                ).span_suggestion(
                    seq_span,
                    "..or a vertical bar to match on multiple alternatives",
                    format!("{}", seq_snippet.replace(",", " |")),
                    Applicability::MachineApplicable
                );
            }
            return Err(err);
        }
        Ok(pat)
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

    /// Parses a pattern, that may be a or-pattern (e.g. `Some(Foo | Bar)`).
    fn parse_pat_with_or(&mut self, expected: Option<&'static str>) -> PResult<'a, P<Pat>> {
        // Parse the first pattern.
        let first_pat = self.parse_pat(expected)?;

        // If the next token is not a `|`, this is not an or-pattern and
        // we should exit here.
        if !self.check(&token::BinOp(token::Or)) {
            return Ok(first_pat)
        }

        let lo = first_pat.span;

        let mut pats = vec![first_pat];

        while self.eat(&token::BinOp(token::Or)) {
            pats.push(self.parse_pat_with_range_pat(
                true, expected
            )?);
        }

        let or_pattern_span = lo.to(self.prev_span);

        self.sess.gated_spans.or_patterns.borrow_mut().push(or_pattern_span);

        Ok(self.mk_pat(or_pattern_span, PatKind::Or(pats)))
    }

    /// Parses a pattern, with a setting whether modern range patterns (e.g., `a..=b`, `a..b` are
    /// allowed).
    fn parse_pat_with_range_pat(
        &mut self,
        allow_range_pat: bool,
        expected: Option<&'static str>,
    ) -> PResult<'a, P<Pat>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);
        maybe_whole!(self, NtPat, |x| x);

        let lo = self.token.span;
        let pat = match self.token.kind {
            token::BinOp(token::And) | token::AndAnd => self.parse_pat_deref(expected)?,
            token::OpenDelim(token::Paren) => self.parse_pat_tuple_or_parens()?,
            token::OpenDelim(token::Bracket) => {
                // Parse `[pat, pat,...]` as a slice pattern.
                PatKind::Slice(self.parse_delim_comma_seq(token::Bracket, |p| p.parse_pat(None))?.0)
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
                self.recover_pat_ident_mut_first()?
            } else if self.eat_keyword(kw::Ref) {
                // Parse ref ident @ pat / ref mut ident @ pat
                let mutbl = self.parse_mutability();
                self.parse_pat_ident(BindingMode::ByRef(mutbl))?
            } else if self.eat_keyword(kw::Box) {
                // Parse `box pat`
                PatKind::Box(self.parse_pat_with_range_pat(false, None)?)
            } else if self.token.is_ident() && !self.token.is_reserved_ident() &&
                      self.parse_as_ident() {
                // Parse `ident @ pat`
                // This can give false positives and parse nullary enums,
                // they are dealt with later in resolve.
                self.parse_pat_ident(BindingMode::ByValue(Mutability::Immutable))?
            } else if self.token.is_path_start() {
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
    fn parse_pat_deref(&mut self, expected: Option<&'static str>) -> PResult<'a, PatKind> {
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
        let (fields, trailing_comma) = self.parse_paren_comma_seq(|p| {
            p.parse_pat_with_or(None)
        })?;

        // Here, `(pat,)` is a tuple pattern.
        // For backward compatibility, `(..)` is a tuple pattern as well.
        Ok(if fields.len() == 1 && !(trailing_comma || fields[0].is_rest()) {
            PatKind::Paren(fields.into_iter().nth(0).unwrap())
        } else {
            PatKind::Tuple(fields)
        })
    }

    /// Recover on `mut ref? ident @ pat` and suggest
    /// that the order of `mut` and `ref` is incorrect.
    fn recover_pat_ident_mut_first(&mut self) -> PResult<'a, PatKind> {
        let mutref_span = self.prev_span.to(self.token.span);
        let binding_mode = if self.eat_keyword(kw::Ref) {
            self.struct_span_err(mutref_span, "the order of `mut` and `ref` is incorrect")
                .span_suggestion(
                    mutref_span,
                    "try switching the order",
                    "ref mut".into(),
                    Applicability::MachineApplicable
                )
                .emit();
            BindingMode::ByRef(Mutability::Mutable)
        } else {
            BindingMode::ByValue(Mutability::Mutable)
        };
        self.parse_pat_ident(binding_mode)
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
        expected: Option<&'static str>,
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

    // Helper function to decide whether to parse as ident binding
    // or to try to do something more complex like range patterns.
    fn parse_as_ident(&mut self) -> bool {
        self.look_ahead(1, |t| match t.kind {
            token::OpenDelim(token::Paren) | token::OpenDelim(token::Brace) |
            token::DotDotDot | token::DotDotEq | token::DotDot |
            token::ModSep | token::Not => false,
            _ => true,
        })
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
        let (fields, _) = self.parse_paren_comma_seq(|p| p.parse_pat_with_or(None))?;
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
            let pat = self.parse_pat_with_or(None)?;
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
