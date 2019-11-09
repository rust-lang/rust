use super::{Parser, PathStyle, PrevTokenKind, TokenType};
use super::item::ParamCfg;

use crate::{maybe_whole, maybe_recover_from_interpolated_ty_qpath};
use crate::ptr::P;
use crate::ast::{self, Ty, TyKind, MutTy, BareFnTy, FunctionRetTy, GenericParam, Lifetime, Ident};
use crate::ast::{TraitBoundModifier, TraitObjectSyntax, GenericBound, GenericBounds, PolyTraitRef};
use crate::ast::{Mutability, AnonConst, Mac};
use crate::token::{self, Token};
use crate::source_map::Span;
use crate::symbol::{kw};

use errors::{PResult, Applicability, pluralize};

/// Returns `true` if `IDENT t` can start a type -- `IDENT::a::b`, `IDENT<u8, u8>`,
/// `IDENT<<u8 as Trait>::AssocTy>`.
///
/// Types can also be of the form `IDENT(u8, u8) -> u8`, however this assumes
/// that `IDENT` is not the ident of a fn trait.
fn can_continue_type_after_non_fn_ident(t: &Token) -> bool {
    t == &token::ModSep || t == &token::Lt ||
    t == &token::BinOp(token::Shl)
}

impl<'a> Parser<'a> {
    /// Parses a type.
    pub fn parse_ty(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(true, true, false)
    }

    /// Parses a type in restricted contexts where `+` is not permitted.
    ///
    /// Example 1: `&'a TYPE`
    ///     `+` is prohibited to maintain operator priority (P(+) < P(&)).
    /// Example 2: `value1 as TYPE + value2`
    ///     `+` is prohibited to avoid interactions with expression grammar.
    pub(super) fn parse_ty_no_plus(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(false, true, false)
    }

    /// Parses an optional return type `[ -> TY ]` in a function declaration.
    pub(super) fn parse_ret_ty(&mut self, allow_plus: bool) -> PResult<'a, FunctionRetTy> {
        if self.eat(&token::RArrow) {
            Ok(FunctionRetTy::Ty(self.parse_ty_common(allow_plus, true, false)?))
        } else {
            Ok(FunctionRetTy::Default(self.token.span.shrink_to_lo()))
        }
    }

    pub(super) fn parse_ty_common(&mut self, allow_plus: bool, allow_qpath_recovery: bool,
                       allow_c_variadic: bool) -> PResult<'a, P<Ty>> {
        maybe_recover_from_interpolated_ty_qpath!(self, allow_qpath_recovery);
        maybe_whole!(self, NtTy, |x| x);

        let lo = self.token.span;
        let mut impl_dyn_multi = false;
        let kind = if self.eat(&token::OpenDelim(token::Paren)) {
            // `(TYPE)` is a parenthesized type.
            // `(TYPE,)` is a tuple with a single field of type TYPE.
            let mut ts = vec![];
            let mut last_comma = false;
            while self.token != token::CloseDelim(token::Paren) {
                ts.push(self.parse_ty()?);
                if self.eat(&token::Comma) {
                    last_comma = true;
                } else {
                    last_comma = false;
                    break;
                }
            }
            let trailing_plus = self.prev_token_kind == PrevTokenKind::Plus;
            self.expect(&token::CloseDelim(token::Paren))?;

            if ts.len() == 1 && !last_comma {
                let ty = ts.into_iter().nth(0).unwrap().into_inner();
                let maybe_bounds = allow_plus && self.token.is_like_plus();
                match ty.kind {
                    // `(TY_BOUND_NOPAREN) + BOUND + ...`.
                    TyKind::Path(None, ref path) if maybe_bounds => {
                        self.parse_remaining_bounds(Vec::new(), path.clone(), lo, true)?
                    }
                    TyKind::TraitObject(ref bounds, TraitObjectSyntax::None)
                            if maybe_bounds && bounds.len() == 1 && !trailing_plus => {
                        let path = match bounds[0] {
                            GenericBound::Trait(ref pt, ..) => pt.trait_ref.path.clone(),
                            GenericBound::Outlives(..) => self.bug("unexpected lifetime bound"),
                        };
                        self.parse_remaining_bounds(Vec::new(), path, lo, true)?
                    }
                    // `(TYPE)`
                    _ => TyKind::Paren(P(ty))
                }
            } else {
                TyKind::Tup(ts)
            }
        } else if self.eat(&token::Not) {
            // Never type `!`
            TyKind::Never
        } else if self.eat(&token::BinOp(token::Star)) {
            // Raw pointer
            TyKind::Ptr(self.parse_ptr()?)
        } else if self.eat(&token::OpenDelim(token::Bracket)) {
            // Array or slice
            let t = self.parse_ty()?;
            // Parse optional `; EXPR` in `[TYPE; EXPR]`
            let t = match self.maybe_parse_fixed_length_of_vec()? {
                None => TyKind::Slice(t),
                Some(length) => TyKind::Array(t, AnonConst {
                    id: ast::DUMMY_NODE_ID,
                    value: length,
                }),
            };
            self.expect(&token::CloseDelim(token::Bracket))?;
            t
        } else if self.check(&token::BinOp(token::And)) || self.check(&token::AndAnd) {
            // Reference
            self.expect_and()?;
            self.parse_borrowed_pointee()?
        } else if self.eat_keyword_noexpect(kw::Typeof) {
            // `typeof(EXPR)`
            // In order to not be ambiguous, the type must be surrounded by parens.
            self.expect(&token::OpenDelim(token::Paren))?;
            let e = AnonConst {
                id: ast::DUMMY_NODE_ID,
                value: self.parse_expr()?,
            };
            self.expect(&token::CloseDelim(token::Paren))?;
            TyKind::Typeof(e)
        } else if self.eat_keyword(kw::Underscore) {
            // A type to be inferred `_`
            TyKind::Infer
        } else if self.token_is_bare_fn_keyword() {
            // Function pointer type
            self.parse_ty_bare_fn(Vec::new())?
        } else if self.check_keyword(kw::For) {
            // Function pointer type or bound list (trait object type) starting with a poly-trait.
            //   `for<'lt> [unsafe] [extern "ABI"] fn (&'lt S) -> T`
            //   `for<'lt> Trait1<'lt> + Trait2 + 'a`
            let lo = self.token.span;
            let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
            if self.token_is_bare_fn_keyword() {
                self.parse_ty_bare_fn(lifetime_defs)?
            } else {
                let path = self.parse_path(PathStyle::Type)?;
                let parse_plus = allow_plus && self.check_plus();
                self.parse_remaining_bounds(lifetime_defs, path, lo, parse_plus)?
            }
        } else if self.eat_keyword(kw::Impl) {
            // Always parse bounds greedily for better error recovery.
            let bounds = self.parse_generic_bounds(None)?;
            impl_dyn_multi = bounds.len() > 1 || self.prev_token_kind == PrevTokenKind::Plus;
            TyKind::ImplTrait(ast::DUMMY_NODE_ID, bounds)
        } else if self.check_keyword(kw::Dyn) &&
                  (self.token.span.rust_2018() ||
                   self.look_ahead(1, |t| t.can_begin_bound() &&
                                          !can_continue_type_after_non_fn_ident(t))) {
            self.bump(); // `dyn`
            // Always parse bounds greedily for better error recovery.
            let bounds = self.parse_generic_bounds(None)?;
            impl_dyn_multi = bounds.len() > 1 || self.prev_token_kind == PrevTokenKind::Plus;
            TyKind::TraitObject(bounds, TraitObjectSyntax::Dyn)
        } else if self.check(&token::Question) ||
                  self.check_lifetime() && self.look_ahead(1, |t| t.is_like_plus()) {
            // Bound list (trait object type)
            TyKind::TraitObject(self.parse_generic_bounds_common(allow_plus, None)?,
                                TraitObjectSyntax::None)
        } else if self.eat_lt() {
            // Qualified path
            let (qself, path) = self.parse_qpath(PathStyle::Type)?;
            TyKind::Path(Some(qself), path)
        } else if self.token.is_path_start() {
            // Simple path
            let path = self.parse_path(PathStyle::Type)?;
            if self.eat(&token::Not) {
                // Macro invocation in type position
                let (delim, tts) = self.expect_delimited_token_tree()?;
                let mac = Mac {
                    path,
                    tts,
                    delim,
                    span: lo.to(self.prev_span),
                    prior_type_ascription: self.last_type_ascription,
                };
                TyKind::Mac(mac)
            } else {
                // Just a type path or bound list (trait object type) starting with a trait.
                //   `Type`
                //   `Trait1 + Trait2 + 'a`
                if allow_plus && self.check_plus() {
                    self.parse_remaining_bounds(Vec::new(), path, lo, true)?
                } else {
                    TyKind::Path(None, path)
                }
            }
        } else if self.check(&token::DotDotDot) {
            if allow_c_variadic {
                self.eat(&token::DotDotDot);
                TyKind::CVarArgs
            } else {
                return Err(struct_span_fatal!(
                    self.sess.span_diagnostic,
                    self.token.span,
                    E0743,
                    "only foreign functions are allowed to be C-variadic",
                ));
            }
        } else {
            let msg = format!("expected type, found {}", self.this_token_descr());
            let mut err = self.fatal(&msg);
            err.span_label(self.token.span, "expected type");
            self.maybe_annotate_with_ascription(&mut err, true);
            return Err(err);
        };

        let span = lo.to(self.prev_span);
        let ty = self.mk_ty(span, kind);

        // Try to recover from use of `+` with incorrect priority.
        self.maybe_report_ambiguous_plus(allow_plus, impl_dyn_multi, &ty);
        self.maybe_recover_from_bad_type_plus(allow_plus, &ty)?;
        self.maybe_recover_from_bad_qpath(ty, allow_qpath_recovery)
    }

    fn parse_remaining_bounds(&mut self, generic_params: Vec<GenericParam>, path: ast::Path,
                              lo: Span, parse_plus: bool) -> PResult<'a, TyKind> {
        let poly_trait_ref = PolyTraitRef::new(generic_params, path, lo.to(self.prev_span));
        let mut bounds = vec![GenericBound::Trait(poly_trait_ref, TraitBoundModifier::None)];
        if parse_plus {
            self.eat_plus(); // `+`, or `+=` gets split and `+` is discarded
            bounds.append(&mut self.parse_generic_bounds(Some(self.prev_span))?);
        }
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::None))
    }

    fn parse_ptr(&mut self) -> PResult<'a, MutTy> {
        let mutbl = self.parse_const_or_mut().unwrap_or_else(|| {
            let span = self.prev_span;
            let msg = "expected mut or const in raw pointer type";
            self.struct_span_err(span, msg)
                .span_label(span, msg)
                .help("use `*mut T` or `*const T` as appropriate")
                .emit();
            Mutability::Immutable
        });
        let t = self.parse_ty_no_plus()?;
        Ok(MutTy { ty: t, mutbl })
    }

    fn maybe_parse_fixed_length_of_vec(&mut self) -> PResult<'a, Option<P<ast::Expr>>> {
        if self.eat(&token::Semi) {
            Ok(Some(self.parse_expr()?))
        } else {
            Ok(None)
        }
    }

    fn parse_borrowed_pointee(&mut self) -> PResult<'a, TyKind> {
        let opt_lifetime = if self.check_lifetime() { Some(self.expect_lifetime()) } else { None };
        let mutbl = self.parse_mutability();
        let ty = self.parse_ty_no_plus()?;
        return Ok(TyKind::Rptr(opt_lifetime, MutTy { ty, mutbl }));
    }

    /// Is the current token one of the keywords that signals a bare function type?
    fn token_is_bare_fn_keyword(&mut self) -> bool {
        self.check_keyword(kw::Fn) ||
            self.check_keyword(kw::Unsafe) ||
            self.check_keyword(kw::Extern)
    }

    /// Parses a `TyKind::BareFn` type.
    fn parse_ty_bare_fn(&mut self, generic_params: Vec<GenericParam>) -> PResult<'a, TyKind> {
        /*

        [unsafe] [extern "ABI"] fn (S) -> T
         ^~~~^           ^~~~^     ^~^    ^
           |               |        |     |
           |               |        |   Return type
           |               |      Argument types
           |               |
           |              ABI
        Function Style
        */

        let unsafety = self.parse_unsafety();
        let abi = self.parse_extern_abi()?;
        self.expect_keyword(kw::Fn)?;
        let cfg = ParamCfg {
            is_self_allowed: false,
            allow_c_variadic: true,
            is_name_required: |_| false,
        };
        let decl = self.parse_fn_decl(cfg, false)?;
        Ok(TyKind::BareFn(P(BareFnTy {
            abi,
            unsafety,
            generic_params,
            decl,
        })))
    }

    pub(super) fn parse_generic_bounds(&mut self,
                                  colon_span: Option<Span>) -> PResult<'a, GenericBounds> {
        self.parse_generic_bounds_common(true, colon_span)
    }

    /// Parses bounds of a type parameter `BOUND + BOUND + ...`, possibly with trailing `+`.
    ///
    /// ```
    /// BOUND = TY_BOUND | LT_BOUND
    /// LT_BOUND = LIFETIME (e.g., `'a`)
    /// TY_BOUND = TY_BOUND_NOPAREN | (TY_BOUND_NOPAREN)
    /// TY_BOUND_NOPAREN = [?] [for<LT_PARAM_DEFS>] SIMPLE_PATH (e.g., `?for<'a: 'b> m::Trait<'a>`)
    /// ```
    fn parse_generic_bounds_common(&mut self,
                                   allow_plus: bool,
                                   colon_span: Option<Span>) -> PResult<'a, GenericBounds> {
        let mut bounds = Vec::new();
        let mut negative_bounds = Vec::new();
        let mut last_plus_span = None;
        let mut was_negative = false;
        loop {
            // This needs to be synchronized with `TokenKind::can_begin_bound`.
            let is_bound_start = self.check_path() || self.check_lifetime() ||
                                 self.check(&token::Not) || // used for error reporting only
                                 self.check(&token::Question) ||
                                 self.check_keyword(kw::For) ||
                                 self.check(&token::OpenDelim(token::Paren));
            if is_bound_start {
                let lo = self.token.span;
                let has_parens = self.eat(&token::OpenDelim(token::Paren));
                let inner_lo = self.token.span;
                let is_negative = self.eat(&token::Not);
                let question = if self.eat(&token::Question) { Some(self.prev_span) } else { None };
                if self.token.is_lifetime() {
                    if let Some(question_span) = question {
                        self.span_err(question_span,
                                      "`?` may only modify trait bounds, not lifetime bounds");
                    }
                    bounds.push(GenericBound::Outlives(self.expect_lifetime()));
                    if has_parens {
                        let inner_span = inner_lo.to(self.prev_span);
                        self.expect(&token::CloseDelim(token::Paren))?;
                        let mut err = self.struct_span_err(
                            lo.to(self.prev_span),
                            "parenthesized lifetime bounds are not supported"
                        );
                        if let Ok(snippet) = self.span_to_snippet(inner_span) {
                            err.span_suggestion_short(
                                lo.to(self.prev_span),
                                "remove the parentheses",
                                snippet.to_owned(),
                                Applicability::MachineApplicable
                            );
                        }
                        err.emit();
                    }
                } else {
                    let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
                    let path = self.parse_path(PathStyle::Type)?;
                    if has_parens {
                        self.expect(&token::CloseDelim(token::Paren))?;
                    }
                    let poly_span = lo.to(self.prev_span);
                    if is_negative {
                        was_negative = true;
                        if let Some(sp) = last_plus_span.or(colon_span) {
                            negative_bounds.push(sp.to(poly_span));
                        }
                    } else {
                        let poly_trait = PolyTraitRef::new(lifetime_defs, path, poly_span);
                        let modifier = if question.is_some() {
                            TraitBoundModifier::Maybe
                        } else {
                            TraitBoundModifier::None
                        };
                        bounds.push(GenericBound::Trait(poly_trait, modifier));
                    }
                }
            } else {
                break
            }

            if !allow_plus || !self.eat_plus() {
                break
            } else {
                last_plus_span = Some(self.prev_span);
            }
        }

        if !negative_bounds.is_empty() || was_negative {
            let negative_bounds_len = negative_bounds.len();
            let last_span = negative_bounds.last().map(|sp| *sp);
            let mut err = self.struct_span_err(
                negative_bounds,
                "negative trait bounds are not supported",
            );
            if let Some(sp) = last_span {
                err.span_label(sp, "negative trait bounds are not supported");
            }
            if let Some(bound_list) = colon_span {
                let bound_list = bound_list.to(self.prev_span);
                let mut new_bound_list = String::new();
                if !bounds.is_empty() {
                    let mut snippets = bounds.iter().map(|bound| bound.span())
                        .map(|span| self.span_to_snippet(span));
                    while let Some(Ok(snippet)) = snippets.next() {
                        new_bound_list.push_str(" + ");
                        new_bound_list.push_str(&snippet);
                    }
                    new_bound_list = new_bound_list.replacen(" +", ":", 1);
                }
                err.span_suggestion_hidden(
                    bound_list,
                    &format!("remove the trait bound{}", pluralize!(negative_bounds_len)),
                    new_bound_list,
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        }

        return Ok(bounds);
    }

    pub(super) fn parse_late_bound_lifetime_defs(&mut self) -> PResult<'a, Vec<GenericParam>> {
        if self.eat_keyword(kw::For) {
            self.expect_lt()?;
            let params = self.parse_generic_params()?;
            self.expect_gt()?;
            // We rely on AST validation to rule out invalid cases: There must not be type
            // parameters, and the lifetime parameters must not have bounds.
            Ok(params)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn check_lifetime(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Lifetime);
        self.token.is_lifetime()
    }

    /// Parses a single lifetime `'a` or panics.
    pub fn expect_lifetime(&mut self) -> Lifetime {
        if let Some(ident) = self.token.lifetime() {
            let span = self.token.span;
            self.bump();
            Lifetime { ident: Ident::new(ident.name, span), id: ast::DUMMY_NODE_ID }
        } else {
            self.span_bug(self.token.span, "not a lifetime")
        }
    }

    pub(super) fn mk_ty(&self, span: Span, kind: TyKind) -> P<Ty> {
        P(Ty { kind, span, id: ast::DUMMY_NODE_ID })
    }
}
