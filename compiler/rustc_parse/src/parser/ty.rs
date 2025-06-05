use rustc_ast::ptr::P;
use rustc_ast::token::{self, IdentIsRaw, MetaVarKind, Token, TokenKind};
use rustc_ast::util::case::Case;
use rustc_ast::{
    self as ast, BareFnTy, BoundAsyncness, BoundConstness, BoundPolarity, DUMMY_NODE_ID, FnRetTy,
    GenericBound, GenericBounds, GenericParam, Generics, Lifetime, MacCall, MutTy, Mutability,
    Pinnedness, PolyTraitRef, PreciseCapturingArg, TraitBoundModifiers, TraitObjectSyntax, Ty,
    TyKind, UnsafeBinderTy,
};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{Applicability, Diag, PResult};
use rustc_span::{ErrorGuaranteed, Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use super::{Parser, PathStyle, SeqSep, TokenType, Trailing};
use crate::errors::{
    self, DynAfterMut, ExpectedFnPathFoundFnKeyword, ExpectedMutOrConstInRawPointerType,
    FnPointerCannotBeAsync, FnPointerCannotBeConst, FnPtrWithGenerics, FnPtrWithGenericsSugg,
    HelpUseLatestEdition, InvalidDynKeyword, LifetimeAfterMut, NeedPlusAfterTraitObjectLifetime,
    NestedCVariadicType, ReturnTypesUseThinArrow,
};
use crate::{exp, maybe_recover_from_interpolated_ty_qpath};

/// Signals whether parsing a type should allow `+`.
///
/// For example, let T be the type `impl Default + 'static`
/// With `AllowPlus::Yes`, T will be parsed successfully
/// With `AllowPlus::No`, parsing T will return a parse error
#[derive(Copy, Clone, PartialEq)]
pub(super) enum AllowPlus {
    Yes,
    No,
}

#[derive(PartialEq)]
pub(super) enum RecoverQPath {
    Yes,
    No,
}

pub(super) enum RecoverQuestionMark {
    Yes,
    No,
}

/// Signals whether parsing a type should recover `->`.
///
/// More specifically, when parsing a function like:
/// ```compile_fail
/// fn foo() => u8 { 0 }
/// fn bar(): u8 { 0 }
/// ```
/// The compiler will try to recover interpreting `foo() => u8` as `foo() -> u8` when calling
/// `parse_ty` with anything except `RecoverReturnSign::No`, and it will try to recover `bar(): u8`
/// as `bar() -> u8` when passing `RecoverReturnSign::Yes` to `parse_ty`
#[derive(Copy, Clone, PartialEq)]
pub(super) enum RecoverReturnSign {
    Yes,
    OnlyFatArrow,
    No,
}

impl RecoverReturnSign {
    /// [RecoverReturnSign::Yes] allows for recovering `fn foo() => u8` and `fn foo(): u8`,
    /// [RecoverReturnSign::OnlyFatArrow] allows for recovering only `fn foo() => u8` (recovering
    /// colons can cause problems when parsing where clauses), and
    /// [RecoverReturnSign::No] doesn't allow for any recovery of the return type arrow
    fn can_recover(self, token: &TokenKind) -> bool {
        match self {
            Self::Yes => matches!(token, token::FatArrow | token::Colon),
            Self::OnlyFatArrow => matches!(token, token::FatArrow),
            Self::No => false,
        }
    }
}

// Is `...` (`CVarArgs`) legal at this level of type parsing?
#[derive(PartialEq)]
enum AllowCVariadic {
    Yes,
    No,
}

/// Returns `true` if `IDENT t` can start a type -- `IDENT::a::b`, `IDENT<u8, u8>`,
/// `IDENT<<u8 as Trait>::AssocTy>`.
///
/// Types can also be of the form `IDENT(u8, u8) -> u8`, however this assumes
/// that `IDENT` is not the ident of a fn trait.
fn can_continue_type_after_non_fn_ident(t: &Token) -> bool {
    t == &token::PathSep || t == &token::Lt || t == &token::Shl
}

fn can_begin_dyn_bound_in_edition_2015(t: &Token) -> bool {
    // `Not`, `Tilde` & `Const` are deliberately not part of this list to
    // contain the number of potential regressions esp. in MBE code.
    // `Const` would regress `rfc-2632-const-trait-impl/mbe-dyn-const-2015.rs`.
    // `Not` would regress `dyn!(...)` macro calls in Rust 2015.
    t.is_path_start()
        || t.is_lifetime()
        || t == &TokenKind::Question
        || t.is_keyword(kw::For)
        || t == &TokenKind::OpenParen
}

impl<'a> Parser<'a> {
    /// Parses a type.
    pub fn parse_ty(&mut self) -> PResult<'a, P<Ty>> {
        // Make sure deeply nested types don't overflow the stack.
        ensure_sufficient_stack(|| {
            self.parse_ty_common(
                AllowPlus::Yes,
                AllowCVariadic::No,
                RecoverQPath::Yes,
                RecoverReturnSign::Yes,
                None,
                RecoverQuestionMark::Yes,
            )
        })
    }

    pub(super) fn parse_ty_with_generics_recovery(
        &mut self,
        ty_params: &Generics,
    ) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::Yes,
            AllowCVariadic::No,
            RecoverQPath::Yes,
            RecoverReturnSign::Yes,
            Some(ty_params),
            RecoverQuestionMark::Yes,
        )
    }

    /// Parse a type suitable for a function or function pointer parameter.
    /// The difference from `parse_ty` is that this version allows `...`
    /// (`CVarArgs`) at the top level of the type.
    pub(super) fn parse_ty_for_param(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::Yes,
            AllowCVariadic::Yes,
            RecoverQPath::Yes,
            RecoverReturnSign::Yes,
            None,
            RecoverQuestionMark::Yes,
        )
    }

    /// Parses a type in restricted contexts where `+` is not permitted.
    ///
    /// Example 1: `&'a TYPE`
    ///     `+` is prohibited to maintain operator priority (P(+) < P(&)).
    /// Example 2: `value1 as TYPE + value2`
    ///     `+` is prohibited to avoid interactions with expression grammar.
    pub(super) fn parse_ty_no_plus(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::No,
            AllowCVariadic::No,
            RecoverQPath::Yes,
            RecoverReturnSign::Yes,
            None,
            RecoverQuestionMark::Yes,
        )
    }

    /// Parses a type following an `as` cast. Similar to `parse_ty_no_plus`, but signaling origin
    /// for better diagnostics involving `?`.
    pub(super) fn parse_as_cast_ty(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::No,
            AllowCVariadic::No,
            RecoverQPath::Yes,
            RecoverReturnSign::Yes,
            None,
            RecoverQuestionMark::No,
        )
    }

    pub(super) fn parse_ty_no_question_mark_recover(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::Yes,
            AllowCVariadic::No,
            RecoverQPath::Yes,
            RecoverReturnSign::Yes,
            None,
            RecoverQuestionMark::No,
        )
    }

    /// Parse a type without recovering `:` as `->` to avoid breaking code such
    /// as `where fn() : for<'a>`.
    pub(super) fn parse_ty_for_where_clause(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::Yes,
            AllowCVariadic::No,
            RecoverQPath::Yes,
            RecoverReturnSign::OnlyFatArrow,
            None,
            RecoverQuestionMark::Yes,
        )
    }

    /// Parses an optional return type `[ -> TY ]` in a function declaration.
    pub(super) fn parse_ret_ty(
        &mut self,
        allow_plus: AllowPlus,
        recover_qpath: RecoverQPath,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, FnRetTy> {
        let lo = self.prev_token.span;
        Ok(if self.eat(exp!(RArrow)) {
            // FIXME(Centril): Can we unconditionally `allow_plus`?
            let ty = self.parse_ty_common(
                allow_plus,
                AllowCVariadic::No,
                recover_qpath,
                recover_return_sign,
                None,
                RecoverQuestionMark::Yes,
            )?;
            FnRetTy::Ty(ty)
        } else if recover_return_sign.can_recover(&self.token.kind) {
            // Don't `eat` to prevent `=>` from being added as an expected token which isn't
            // actually expected and could only confuse users
            self.bump();
            self.dcx().emit_err(ReturnTypesUseThinArrow {
                span: self.prev_token.span,
                suggestion: lo.between(self.token.span),
            });
            let ty = self.parse_ty_common(
                allow_plus,
                AllowCVariadic::No,
                recover_qpath,
                recover_return_sign,
                None,
                RecoverQuestionMark::Yes,
            )?;
            FnRetTy::Ty(ty)
        } else {
            FnRetTy::Default(self.prev_token.span.shrink_to_hi())
        })
    }

    fn parse_ty_common(
        &mut self,
        allow_plus: AllowPlus,
        allow_c_variadic: AllowCVariadic,
        recover_qpath: RecoverQPath,
        recover_return_sign: RecoverReturnSign,
        ty_generics: Option<&Generics>,
        recover_question_mark: RecoverQuestionMark,
    ) -> PResult<'a, P<Ty>> {
        let allow_qpath_recovery = recover_qpath == RecoverQPath::Yes;
        maybe_recover_from_interpolated_ty_qpath!(self, allow_qpath_recovery);

        if let Some(ty) = self.eat_metavar_seq_with_matcher(
            |mv_kind| matches!(mv_kind, MetaVarKind::Ty { .. }),
            |this| this.parse_ty_no_question_mark_recover(),
        ) {
            return Ok(ty);
        }

        let lo = self.token.span;
        let mut impl_dyn_multi = false;
        let kind = if self.check(exp!(OpenParen)) {
            self.parse_ty_tuple_or_parens(lo, allow_plus)?
        } else if self.eat(exp!(Bang)) {
            // Never type `!`
            TyKind::Never
        } else if self.eat(exp!(Star)) {
            self.parse_ty_ptr()?
        } else if self.eat(exp!(OpenBracket)) {
            self.parse_array_or_slice_ty()?
        } else if self.check(exp!(And)) || self.check(exp!(AndAnd)) {
            // Reference
            self.expect_and()?;
            self.parse_borrowed_pointee()?
        } else if self.eat_keyword_noexpect(kw::Typeof) {
            self.parse_typeof_ty()?
        } else if self.eat_keyword(exp!(Underscore)) {
            // A type to be inferred `_`
            TyKind::Infer
        } else if self.check_fn_front_matter(false, Case::Sensitive) {
            // Function pointer type
            self.parse_ty_bare_fn(lo, ThinVec::new(), None, recover_return_sign)?
        } else if self.check_keyword(exp!(For)) {
            // Function pointer type or bound list (trait object type) starting with a poly-trait.
            //   `for<'lt> [unsafe] [extern "ABI"] fn (&'lt S) -> T`
            //   `for<'lt> Trait1<'lt> + Trait2 + 'a`
            let (lifetime_defs, _) = self.parse_late_bound_lifetime_defs()?;
            if self.check_fn_front_matter(false, Case::Sensitive) {
                self.parse_ty_bare_fn(
                    lo,
                    lifetime_defs,
                    Some(self.prev_token.span.shrink_to_lo()),
                    recover_return_sign,
                )?
            } else {
                // Try to recover `for<'a> dyn Trait` or `for<'a> impl Trait`.
                if self.may_recover()
                    && (self.eat_keyword_noexpect(kw::Impl) || self.eat_keyword_noexpect(kw::Dyn))
                {
                    let kw = self.prev_token.ident().unwrap().0;
                    let removal_span = kw.span.with_hi(self.token.span.lo());
                    let path = self.parse_path(PathStyle::Type)?;
                    let parse_plus = allow_plus == AllowPlus::Yes && self.check_plus();
                    let kind =
                        self.parse_remaining_bounds_path(lifetime_defs, path, lo, parse_plus)?;
                    let err = self.dcx().create_err(errors::TransposeDynOrImpl {
                        span: kw.span,
                        kw: kw.name.as_str(),
                        sugg: errors::TransposeDynOrImplSugg {
                            removal_span,
                            insertion_span: lo.shrink_to_lo(),
                            kw: kw.name.as_str(),
                        },
                    });

                    // Take the parsed bare trait object and turn it either
                    // into a `dyn` object or an `impl Trait`.
                    let kind = match (kind, kw.name) {
                        (TyKind::TraitObject(bounds, _), kw::Dyn) => {
                            TyKind::TraitObject(bounds, TraitObjectSyntax::Dyn)
                        }
                        (TyKind::TraitObject(bounds, _), kw::Impl) => {
                            TyKind::ImplTrait(ast::DUMMY_NODE_ID, bounds)
                        }
                        _ => return Err(err),
                    };
                    err.emit();
                    kind
                } else {
                    let path = self.parse_path(PathStyle::Type)?;
                    let parse_plus = allow_plus == AllowPlus::Yes && self.check_plus();
                    self.parse_remaining_bounds_path(lifetime_defs, path, lo, parse_plus)?
                }
            }
        } else if self.eat_keyword(exp!(Impl)) {
            self.parse_impl_ty(&mut impl_dyn_multi)?
        } else if self.is_explicit_dyn_type() {
            self.parse_dyn_ty(&mut impl_dyn_multi)?
        } else if self.eat_lt() {
            // Qualified path
            let (qself, path) = self.parse_qpath(PathStyle::Type)?;
            TyKind::Path(Some(qself), path)
        } else if self.check_path() {
            self.parse_path_start_ty(lo, allow_plus, ty_generics)?
        } else if self.can_begin_bound() {
            self.parse_bare_trait_object(lo, allow_plus)?
        } else if self.eat(exp!(DotDotDot)) {
            match allow_c_variadic {
                AllowCVariadic::Yes => TyKind::CVarArgs,
                AllowCVariadic::No => {
                    // FIXME(c_variadic): Should we just allow `...` syntactically
                    // anywhere in a type and use semantic restrictions instead?
                    // NOTE: This may regress certain MBE calls if done incorrectly.
                    let guar = self.dcx().emit_err(NestedCVariadicType { span: lo });
                    TyKind::Err(guar)
                }
            }
        } else if self.check_keyword(exp!(Unsafe))
            && self.look_ahead(1, |tok| tok.kind == token::Lt)
        {
            self.parse_unsafe_binder_ty()?
        } else {
            let msg = format!("expected type, found {}", super::token_descr(&self.token));
            let mut err = self.dcx().struct_span_err(lo, msg);
            err.span_label(lo, "expected type");
            return Err(err);
        };

        let span = lo.to(self.prev_token.span);
        let mut ty = self.mk_ty(span, kind);

        // Try to recover from use of `+` with incorrect priority.
        match allow_plus {
            AllowPlus::Yes => self.maybe_recover_from_bad_type_plus(&ty)?,
            AllowPlus::No => self.maybe_report_ambiguous_plus(impl_dyn_multi, &ty),
        }
        if let RecoverQuestionMark::Yes = recover_question_mark {
            ty = self.maybe_recover_from_question_mark(ty);
        }
        if allow_qpath_recovery { self.maybe_recover_from_bad_qpath(ty) } else { Ok(ty) }
    }

    fn parse_unsafe_binder_ty(&mut self) -> PResult<'a, TyKind> {
        let lo = self.token.span;
        assert!(self.eat_keyword(exp!(Unsafe)));
        self.expect_lt()?;
        let generic_params = self.parse_generic_params()?;
        self.expect_gt()?;
        let inner_ty = self.parse_ty()?;
        let span = lo.to(self.prev_token.span);
        self.psess.gated_spans.gate(sym::unsafe_binders, span);

        Ok(TyKind::UnsafeBinder(P(UnsafeBinderTy { generic_params, inner_ty })))
    }

    /// Parses either:
    /// - `(TYPE)`, a parenthesized type.
    /// - `(TYPE,)`, a tuple with a single field of type TYPE.
    fn parse_ty_tuple_or_parens(&mut self, lo: Span, allow_plus: AllowPlus) -> PResult<'a, TyKind> {
        let mut trailing_plus = false;
        let (ts, trailing) = self.parse_paren_comma_seq(|p| {
            let ty = p.parse_ty()?;
            trailing_plus = p.prev_token == TokenKind::Plus;
            Ok(ty)
        })?;

        if ts.len() == 1 && matches!(trailing, Trailing::No) {
            let ty = ts.into_iter().next().unwrap().into_inner();
            let maybe_bounds = allow_plus == AllowPlus::Yes && self.token.is_like_plus();
            match ty.kind {
                // `(TY_BOUND_NOPAREN) + BOUND + ...`.
                TyKind::Path(None, path) if maybe_bounds => {
                    self.parse_remaining_bounds_path(ThinVec::new(), path, lo, true)
                }
                // For `('a) + …`, we know that `'a` in type position already lead to an error being
                // emitted. To reduce output, let's indirectly suppress E0178 (bad `+` in type) and
                // other irrelevant consequential errors.
                TyKind::TraitObject(bounds, TraitObjectSyntax::None)
                    if maybe_bounds && bounds.len() == 1 && !trailing_plus =>
                {
                    self.parse_remaining_bounds(bounds, true)
                }
                // `(TYPE)`
                _ => Ok(TyKind::Paren(P(ty))),
            }
        } else {
            Ok(TyKind::Tup(ts))
        }
    }

    fn parse_bare_trait_object(&mut self, lo: Span, allow_plus: AllowPlus) -> PResult<'a, TyKind> {
        // A lifetime only begins a bare trait object type if it is followed by `+`!
        if self.token.is_lifetime() && !self.look_ahead(1, |t| t.is_like_plus()) {
            // In Rust 2021 and beyond, we assume that the user didn't intend to write a bare trait
            // object type with a leading lifetime bound since that seems very unlikely given the
            // fact that `dyn`-less trait objects are *semantically* invalid.
            if self.psess.edition.at_least_rust_2021() {
                let lt = self.expect_lifetime();
                let mut err = self.dcx().struct_span_err(lo, "expected type, found lifetime");
                err.span_label(lo, "expected type");
                return Ok(match self.maybe_recover_ref_ty_no_leading_ampersand(lt, lo, err) {
                    Ok(ref_ty) => ref_ty,
                    Err(err) => TyKind::Err(err.emit()),
                });
            }

            self.dcx().emit_err(NeedPlusAfterTraitObjectLifetime {
                span: lo,
                suggestion: lo.shrink_to_hi(),
            });
        }
        Ok(TyKind::TraitObject(
            self.parse_generic_bounds_common(allow_plus)?,
            TraitObjectSyntax::None,
        ))
    }

    fn maybe_recover_ref_ty_no_leading_ampersand<'cx>(
        &mut self,
        lt: Lifetime,
        lo: Span,
        mut err: Diag<'cx>,
    ) -> Result<TyKind, Diag<'cx>> {
        if !self.may_recover() {
            return Err(err);
        }
        let snapshot = self.create_snapshot_for_diagnostic();
        let mutbl = self.parse_mutability();
        match self.parse_ty_no_plus() {
            Ok(ty) => {
                err.span_suggestion_verbose(
                    lo.shrink_to_lo(),
                    "you might have meant to write a reference type here",
                    "&",
                    Applicability::MaybeIncorrect,
                );
                err.emit();
                Ok(TyKind::Ref(Some(lt), MutTy { ty, mutbl }))
            }
            Err(diag) => {
                diag.cancel();
                self.restore_snapshot(snapshot);
                Err(err)
            }
        }
    }

    fn parse_remaining_bounds_path(
        &mut self,
        generic_params: ThinVec<GenericParam>,
        path: ast::Path,
        lo: Span,
        parse_plus: bool,
    ) -> PResult<'a, TyKind> {
        let poly_trait_ref = PolyTraitRef::new(
            generic_params,
            path,
            TraitBoundModifiers::NONE,
            lo.to(self.prev_token.span),
        );
        let bounds = vec![GenericBound::Trait(poly_trait_ref)];
        self.parse_remaining_bounds(bounds, parse_plus)
    }

    /// Parse the remainder of a bare trait object type given an already parsed list.
    fn parse_remaining_bounds(
        &mut self,
        mut bounds: GenericBounds,
        plus: bool,
    ) -> PResult<'a, TyKind> {
        if plus {
            self.eat_plus(); // `+`, or `+=` gets split and `+` is discarded
            bounds.append(&mut self.parse_generic_bounds()?);
        }
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::None))
    }

    /// Parses a raw pointer type: `*[const | mut] $type`.
    fn parse_ty_ptr(&mut self) -> PResult<'a, TyKind> {
        let mutbl = self.parse_const_or_mut().unwrap_or_else(|| {
            let span = self.prev_token.span;
            self.dcx().emit_err(ExpectedMutOrConstInRawPointerType {
                span,
                after_asterisk: span.shrink_to_hi(),
            });
            Mutability::Not
        });
        let ty = self.parse_ty_no_plus()?;
        Ok(TyKind::Ptr(MutTy { ty, mutbl }))
    }

    /// Parses an array (`[TYPE; EXPR]`) or slice (`[TYPE]`) type.
    /// The opening `[` bracket is already eaten.
    fn parse_array_or_slice_ty(&mut self) -> PResult<'a, TyKind> {
        let elt_ty = match self.parse_ty() {
            Ok(ty) => ty,
            Err(err)
                if self.look_ahead(1, |t| *t == token::CloseBracket)
                    | self.look_ahead(1, |t| *t == token::Semi) =>
            {
                // Recover from `[LIT; EXPR]` and `[LIT]`
                self.bump();
                let guar = err.emit();
                self.mk_ty(self.prev_token.span, TyKind::Err(guar))
            }
            Err(err) => return Err(err),
        };

        let ty = if self.eat(exp!(Semi)) {
            let mut length = self.parse_expr_anon_const()?;
            if let Err(e) = self.expect(exp!(CloseBracket)) {
                // Try to recover from `X<Y, ...>` when `X::<Y, ...>` works
                self.check_mistyped_turbofish_with_multiple_type_params(e, &mut length.value)?;
                self.expect(exp!(CloseBracket))?;
            }
            TyKind::Array(elt_ty, length)
        } else {
            self.expect(exp!(CloseBracket))?;
            TyKind::Slice(elt_ty)
        };

        Ok(ty)
    }

    fn parse_borrowed_pointee(&mut self) -> PResult<'a, TyKind> {
        let and_span = self.prev_token.span;
        let mut opt_lifetime = self.check_lifetime().then(|| self.expect_lifetime());
        let (pinned, mut mutbl) = match self.parse_pin_and_mut() {
            Some(pin_mut) => pin_mut,
            None => (Pinnedness::Not, self.parse_mutability()),
        };
        if self.token.is_lifetime() && mutbl == Mutability::Mut && opt_lifetime.is_none() {
            // A lifetime is invalid here: it would be part of a bare trait bound, which requires
            // it to be followed by a plus, but we disallow plus in the pointee type.
            // So we can handle this case as an error here, and suggest `'a mut`.
            // If there *is* a plus next though, handling the error later provides better suggestions
            // (like adding parentheses)
            if !self.look_ahead(1, |t| t.is_like_plus()) {
                let lifetime_span = self.token.span;
                let span = and_span.to(lifetime_span);

                let (suggest_lifetime, snippet) =
                    if let Ok(lifetime_src) = self.span_to_snippet(lifetime_span) {
                        (Some(span), lifetime_src)
                    } else {
                        (None, String::new())
                    };
                self.dcx().emit_err(LifetimeAfterMut { span, suggest_lifetime, snippet });

                opt_lifetime = Some(self.expect_lifetime());
            }
        } else if self.token.is_keyword(kw::Dyn)
            && mutbl == Mutability::Not
            && self.look_ahead(1, |t| t.is_keyword(kw::Mut))
        {
            // We have `&dyn mut ...`, which is invalid and should be `&mut dyn ...`.
            let span = and_span.to(self.look_ahead(1, |t| t.span));
            self.dcx().emit_err(DynAfterMut { span });

            // Recovery
            mutbl = Mutability::Mut;
            let (dyn_tok, dyn_tok_sp) = (self.token, self.token_spacing);
            self.bump();
            self.bump_with((dyn_tok, dyn_tok_sp));
        }
        let ty = self.parse_ty_no_plus()?;
        Ok(match pinned {
            Pinnedness::Not => TyKind::Ref(opt_lifetime, MutTy { ty, mutbl }),
            Pinnedness::Pinned => TyKind::PinnedRef(opt_lifetime, MutTy { ty, mutbl }),
        })
    }

    /// Parses `pin` and `mut` annotations on references.
    ///
    /// It must be either `pin const` or `pin mut`.
    pub(crate) fn parse_pin_and_mut(&mut self) -> Option<(Pinnedness, Mutability)> {
        if self.token.is_ident_named(sym::pin) {
            let result = self.look_ahead(1, |token| {
                if token.is_keyword(kw::Const) {
                    Some((Pinnedness::Pinned, Mutability::Not))
                } else if token.is_keyword(kw::Mut) {
                    Some((Pinnedness::Pinned, Mutability::Mut))
                } else {
                    None
                }
            });
            if result.is_some() {
                self.psess.gated_spans.gate(sym::pin_ergonomics, self.token.span);
                self.bump();
                self.bump();
            }
            result
        } else {
            None
        }
    }

    // Parses the `typeof(EXPR)`.
    // To avoid ambiguity, the type is surrounded by parentheses.
    fn parse_typeof_ty(&mut self) -> PResult<'a, TyKind> {
        self.expect(exp!(OpenParen))?;
        let expr = self.parse_expr_anon_const()?;
        self.expect(exp!(CloseParen))?;
        Ok(TyKind::Typeof(expr))
    }

    /// Parses a function pointer type (`TyKind::BareFn`).
    /// ```ignore (illustrative)
    ///    [unsafe] [extern "ABI"] fn (S) -> T
    /// //  ^~~~~^          ^~~~^     ^~^    ^
    /// //    |               |        |     |
    /// //    |               |        |   Return type
    /// // Function Style    ABI  Parameter types
    /// ```
    /// We actually parse `FnHeader FnDecl`, but we error on `const` and `async` qualifiers.
    fn parse_ty_bare_fn(
        &mut self,
        lo: Span,
        mut params: ThinVec<GenericParam>,
        param_insertion_point: Option<Span>,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, TyKind> {
        let inherited_vis = rustc_ast::Visibility {
            span: rustc_span::DUMMY_SP,
            kind: rustc_ast::VisibilityKind::Inherited,
            tokens: None,
        };
        let span_start = self.token.span;
        let ast::FnHeader { ext, safety, constness, coroutine_kind } =
            self.parse_fn_front_matter(&inherited_vis, Case::Sensitive)?;
        let fn_start_lo = self.prev_token.span.lo();
        if self.may_recover() && self.token == TokenKind::Lt {
            self.recover_fn_ptr_with_generics(lo, &mut params, param_insertion_point)?;
        }
        let decl = self.parse_fn_decl(|_| false, AllowPlus::No, recover_return_sign)?;
        let whole_span = lo.to(self.prev_token.span);

        // Order/parsing of "front matter" follows:
        // `<constness> <coroutine_kind> <safety> <extern> fn()`
        //  ^           ^                ^        ^        ^
        //  |           |                |        |        fn_start_lo
        //  |           |                |        ext_sp.lo
        //  |           |                safety_sp.lo
        //  |           coroutine_sp.lo
        //  const_sp.lo
        if let ast::Const::Yes(const_span) = constness {
            let next_token_lo = if let Some(
                ast::CoroutineKind::Async { span, .. }
                | ast::CoroutineKind::Gen { span, .. }
                | ast::CoroutineKind::AsyncGen { span, .. },
            ) = coroutine_kind
            {
                span.lo()
            } else if let ast::Safety::Unsafe(span) | ast::Safety::Safe(span) = safety {
                span.lo()
            } else if let ast::Extern::Implicit(span) | ast::Extern::Explicit(_, span) = ext {
                span.lo()
            } else {
                fn_start_lo
            };
            let sugg_span = const_span.with_hi(next_token_lo);
            self.dcx().emit_err(FnPointerCannotBeConst {
                span: whole_span,
                qualifier: const_span,
                suggestion: sugg_span,
            });
        }
        if let Some(ast::CoroutineKind::Async { span: async_span, .. }) = coroutine_kind {
            let next_token_lo = if let ast::Safety::Unsafe(span) | ast::Safety::Safe(span) = safety
            {
                span.lo()
            } else if let ast::Extern::Implicit(span) | ast::Extern::Explicit(_, span) = ext {
                span.lo()
            } else {
                fn_start_lo
            };
            let sugg_span = async_span.with_hi(next_token_lo);
            self.dcx().emit_err(FnPointerCannotBeAsync {
                span: whole_span,
                qualifier: async_span,
                suggestion: sugg_span,
            });
        }
        // FIXME(gen_blocks): emit a similar error for `gen fn()`
        let decl_span = span_start.to(self.prev_token.span);
        Ok(TyKind::BareFn(P(BareFnTy { ext, safety, generic_params: params, decl, decl_span })))
    }

    /// Recover from function pointer types with a generic parameter list (e.g. `fn<'a>(&'a str)`).
    fn recover_fn_ptr_with_generics(
        &mut self,
        lo: Span,
        params: &mut ThinVec<GenericParam>,
        param_insertion_point: Option<Span>,
    ) -> PResult<'a, ()> {
        let generics = self.parse_generics()?;
        let arity = generics.params.len();

        let mut lifetimes: ThinVec<_> = generics
            .params
            .into_iter()
            .filter(|param| matches!(param.kind, ast::GenericParamKind::Lifetime))
            .collect();

        let sugg = if !lifetimes.is_empty() {
            let snippet =
                lifetimes.iter().map(|param| param.ident.as_str()).intersperse(", ").collect();

            let (left, snippet) = if let Some(span) = param_insertion_point {
                (span, if params.is_empty() { snippet } else { format!(", {snippet}") })
            } else {
                (lo.shrink_to_lo(), format!("for<{snippet}> "))
            };

            Some(FnPtrWithGenericsSugg {
                left,
                snippet,
                right: generics.span,
                arity,
                for_param_list_exists: param_insertion_point.is_some(),
            })
        } else {
            None
        };

        self.dcx().emit_err(FnPtrWithGenerics { span: generics.span, sugg });
        params.append(&mut lifetimes);
        Ok(())
    }

    /// Parses an `impl B0 + ... + Bn` type.
    fn parse_impl_ty(&mut self, impl_dyn_multi: &mut bool) -> PResult<'a, TyKind> {
        if self.token.is_lifetime() {
            self.look_ahead(1, |t| {
                if let token::Ident(sym, _) = t.kind {
                    // parse pattern with "'a Sized" we're supposed to give suggestion like
                    // "'a + Sized"
                    self.dcx().emit_err(errors::MissingPlusBounds {
                        span: self.token.span,
                        hi: self.token.span.shrink_to_hi(),
                        sym,
                    });
                }
            })
        }

        // Always parse bounds greedily for better error recovery.
        let bounds = self.parse_generic_bounds()?;

        *impl_dyn_multi = bounds.len() > 1 || self.prev_token == TokenKind::Plus;

        Ok(TyKind::ImplTrait(ast::DUMMY_NODE_ID, bounds))
    }

    fn parse_precise_capturing_args(
        &mut self,
    ) -> PResult<'a, (ThinVec<PreciseCapturingArg>, Span)> {
        let lo = self.token.span;
        self.expect_lt()?;
        let (args, _, _) = self.parse_seq_to_before_tokens(
            &[exp!(Gt)],
            &[&TokenKind::Ge, &TokenKind::Shr, &TokenKind::Shr],
            SeqSep::trailing_allowed(exp!(Comma)),
            |self_| {
                if self_.check_keyword(exp!(SelfUpper)) {
                    self_.bump();
                    Ok(PreciseCapturingArg::Arg(
                        ast::Path::from_ident(self_.prev_token.ident().unwrap().0),
                        DUMMY_NODE_ID,
                    ))
                } else if self_.check_ident() {
                    Ok(PreciseCapturingArg::Arg(
                        ast::Path::from_ident(self_.parse_ident()?),
                        DUMMY_NODE_ID,
                    ))
                } else if self_.check_lifetime() {
                    Ok(PreciseCapturingArg::Lifetime(self_.expect_lifetime()))
                } else {
                    self_.unexpected_any()
                }
            },
        )?;
        self.expect_gt()?;
        Ok((args, lo.to(self.prev_token.span)))
    }

    /// Is a `dyn B0 + ... + Bn` type allowed here?
    fn is_explicit_dyn_type(&mut self) -> bool {
        self.check_keyword(exp!(Dyn))
            && (self.token_uninterpolated_span().at_least_rust_2018()
                || self.look_ahead(1, |t| {
                    (can_begin_dyn_bound_in_edition_2015(t) || *t == TokenKind::Star)
                        && !can_continue_type_after_non_fn_ident(t)
                }))
    }

    /// Parses a `dyn B0 + ... + Bn` type.
    ///
    /// Note that this does *not* parse bare trait objects.
    fn parse_dyn_ty(&mut self, impl_dyn_multi: &mut bool) -> PResult<'a, TyKind> {
        let lo = self.token.span;
        self.bump(); // `dyn`

        // parse dyn* types
        let syntax = if self.eat(exp!(Star)) {
            self.psess.gated_spans.gate(sym::dyn_star, lo.to(self.prev_token.span));
            TraitObjectSyntax::DynStar
        } else {
            TraitObjectSyntax::Dyn
        };

        // Always parse bounds greedily for better error recovery.
        let bounds = self.parse_generic_bounds()?;
        *impl_dyn_multi = bounds.len() > 1 || self.prev_token == TokenKind::Plus;
        Ok(TyKind::TraitObject(bounds, syntax))
    }

    /// Parses a type starting with a path.
    ///
    /// This can be:
    /// 1. a type macro, `mac!(...)`,
    /// 2. a bare trait object, `B0 + ... + Bn`,
    /// 3. or a path, `path::to::MyType`.
    fn parse_path_start_ty(
        &mut self,
        lo: Span,
        allow_plus: AllowPlus,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, TyKind> {
        // Simple path
        let path = self.parse_path_inner(PathStyle::Type, ty_generics)?;
        if self.eat(exp!(Bang)) {
            // Macro invocation in type position
            Ok(TyKind::MacCall(P(MacCall { path, args: self.parse_delim_args()? })))
        } else if allow_plus == AllowPlus::Yes && self.check_plus() {
            // `Trait1 + Trait2 + 'a`
            self.parse_remaining_bounds_path(ThinVec::new(), path, lo, true)
        } else {
            // Just a type path.
            Ok(TyKind::Path(None, path))
        }
    }

    pub(super) fn parse_generic_bounds(&mut self) -> PResult<'a, GenericBounds> {
        self.parse_generic_bounds_common(AllowPlus::Yes)
    }

    /// Parses bounds of a type parameter `BOUND + BOUND + ...`, possibly with trailing `+`.
    ///
    /// See `parse_generic_bound` for the `BOUND` grammar.
    fn parse_generic_bounds_common(&mut self, allow_plus: AllowPlus) -> PResult<'a, GenericBounds> {
        let mut bounds = Vec::new();

        // In addition to looping while we find generic bounds:
        // We continue even if we find a keyword. This is necessary for error recovery on,
        // for example, `impl fn()`. The only keyword that can go after generic bounds is
        // `where`, so stop if it's it.
        // We also continue if we find types (not traits), again for error recovery.
        while self.can_begin_bound()
            || (self.may_recover()
                && (self.token.can_begin_type()
                    || (self.token.is_reserved_ident() && !self.token.is_keyword(kw::Where))))
        {
            if self.token.is_keyword(kw::Dyn) {
                // Account for `&dyn Trait + dyn Other`.
                self.bump();
                self.dcx().emit_err(InvalidDynKeyword {
                    span: self.prev_token.span,
                    suggestion: self.prev_token.span.until(self.token.span),
                });
            }
            bounds.push(self.parse_generic_bound()?);
            if allow_plus == AllowPlus::No || !self.eat_plus() {
                break;
            }
        }

        Ok(bounds)
    }

    /// Can the current token begin a bound?
    fn can_begin_bound(&mut self) -> bool {
        self.check_path()
            || self.check_lifetime()
            || self.check(exp!(Bang))
            || self.check(exp!(Question))
            || self.check(exp!(Tilde))
            || self.check_keyword(exp!(For))
            || self.check(exp!(OpenParen))
            || self.check_keyword(exp!(Const))
            || self.check_keyword(exp!(Async))
            || self.check_keyword(exp!(Use))
    }

    /// Parses a bound according to the grammar:
    /// ```ebnf
    /// BOUND = TY_BOUND | LT_BOUND
    /// ```
    fn parse_generic_bound(&mut self) -> PResult<'a, GenericBound> {
        let lo = self.token.span;
        let leading_token = self.prev_token;
        let has_parens = self.eat(exp!(OpenParen));

        let bound = if self.token.is_lifetime() {
            self.parse_generic_lt_bound(lo, has_parens)?
        } else if self.eat_keyword(exp!(Use)) {
            // parse precise captures, if any. This is `use<'lt, 'lt, P, P>`; a list of
            // lifetimes and ident params (including SelfUpper). These are validated later
            // for order, duplication, and whether they actually reference params.
            let use_span = self.prev_token.span;
            let (args, args_span) = self.parse_precise_capturing_args()?;
            GenericBound::Use(args, use_span.to(args_span))
        } else {
            self.parse_generic_ty_bound(lo, has_parens, &leading_token)?
        };

        Ok(bound)
    }

    /// Parses a lifetime ("outlives") bound, e.g. `'a`, according to:
    /// ```ebnf
    /// LT_BOUND = LIFETIME
    /// ```
    fn parse_generic_lt_bound(&mut self, lo: Span, has_parens: bool) -> PResult<'a, GenericBound> {
        let lt = self.expect_lifetime();
        let bound = GenericBound::Outlives(lt);
        if has_parens {
            // FIXME(Centril): Consider not erroring here and accepting `('lt)` instead,
            // possibly introducing `GenericBound::Paren(P<GenericBound>)`?
            self.recover_paren_lifetime(lo)?;
        }
        Ok(bound)
    }

    /// Emits an error if any trait bound modifiers were present.
    fn error_lt_bound_with_modifiers(
        &self,
        modifiers: TraitBoundModifiers,
        binder_span: Option<Span>,
    ) -> ErrorGuaranteed {
        let TraitBoundModifiers { constness, asyncness, polarity } = modifiers;

        match constness {
            BoundConstness::Never => {}
            BoundConstness::Always(span) | BoundConstness::Maybe(span) => {
                return self
                    .dcx()
                    .emit_err(errors::ModifierLifetime { span, modifier: constness.as_str() });
            }
        }

        match polarity {
            BoundPolarity::Positive => {}
            BoundPolarity::Negative(span) | BoundPolarity::Maybe(span) => {
                return self
                    .dcx()
                    .emit_err(errors::ModifierLifetime { span, modifier: polarity.as_str() });
            }
        }

        match asyncness {
            BoundAsyncness::Normal => {}
            BoundAsyncness::Async(span) => {
                return self
                    .dcx()
                    .emit_err(errors::ModifierLifetime { span, modifier: asyncness.as_str() });
            }
        }

        if let Some(span) = binder_span {
            return self.dcx().emit_err(errors::ModifierLifetime { span, modifier: "for<...>" });
        }

        unreachable!("lifetime bound intercepted in `parse_generic_ty_bound` but no modifiers?")
    }

    /// Recover on `('lifetime)` with `(` already eaten.
    fn recover_paren_lifetime(&mut self, lo: Span) -> PResult<'a, ()> {
        self.expect(exp!(CloseParen))?;
        let span = lo.to(self.prev_token.span);
        let sugg = errors::RemoveParens { lo, hi: self.prev_token.span };

        self.dcx().emit_err(errors::ParenthesizedLifetime { span, sugg });
        Ok(())
    }

    /// Parses the modifiers that may precede a trait in a bound, e.g. `?Trait` or `~const Trait`.
    ///
    /// If no modifiers are present, this does not consume any tokens.
    ///
    /// ```ebnf
    /// CONSTNESS = [["~"] "const"]
    /// ASYNCNESS = ["async"]
    /// POLARITY = ["?" | "!"]
    /// ```
    ///
    /// See `parse_generic_ty_bound` for the complete grammar of trait bound modifiers.
    fn parse_trait_bound_modifiers(&mut self) -> PResult<'a, TraitBoundModifiers> {
        let modifier_lo = self.token.span;
        let constness = if self.eat(exp!(Tilde)) {
            let tilde = self.prev_token.span;
            self.expect_keyword(exp!(Const))?;
            let span = tilde.to(self.prev_token.span);
            self.psess.gated_spans.gate(sym::const_trait_impl, span);
            BoundConstness::Maybe(span)
        } else if self.eat_keyword(exp!(Const)) {
            self.psess.gated_spans.gate(sym::const_trait_impl, self.prev_token.span);
            BoundConstness::Always(self.prev_token.span)
        } else {
            BoundConstness::Never
        };

        let asyncness = if self.token_uninterpolated_span().at_least_rust_2018()
            && self.eat_keyword(exp!(Async))
        {
            self.psess.gated_spans.gate(sym::async_trait_bounds, self.prev_token.span);
            BoundAsyncness::Async(self.prev_token.span)
        } else if self.may_recover()
            && self.token_uninterpolated_span().is_rust_2015()
            && self.is_kw_followed_by_ident(kw::Async)
        {
            self.bump(); // eat `async`
            self.dcx().emit_err(errors::AsyncBoundModifierIn2015 {
                span: self.prev_token.span,
                help: HelpUseLatestEdition::new(),
            });
            self.psess.gated_spans.gate(sym::async_trait_bounds, self.prev_token.span);
            BoundAsyncness::Async(self.prev_token.span)
        } else {
            BoundAsyncness::Normal
        };
        let modifier_hi = self.prev_token.span;

        let polarity = if self.eat(exp!(Question)) {
            BoundPolarity::Maybe(self.prev_token.span)
        } else if self.eat(exp!(Bang)) {
            self.psess.gated_spans.gate(sym::negative_bounds, self.prev_token.span);
            BoundPolarity::Negative(self.prev_token.span)
        } else {
            BoundPolarity::Positive
        };

        // Enforce the mutual-exclusivity of `const`/`async` and `?`/`!`.
        match polarity {
            BoundPolarity::Positive => {
                // All trait bound modifiers allowed to combine with positive polarity
            }
            BoundPolarity::Maybe(polarity_span) | BoundPolarity::Negative(polarity_span) => {
                match (asyncness, constness) {
                    (BoundAsyncness::Normal, BoundConstness::Never) => {
                        // Ok, no modifiers.
                    }
                    (_, _) => {
                        let constness = constness.as_str();
                        let asyncness = asyncness.as_str();
                        let glue =
                            if !constness.is_empty() && !asyncness.is_empty() { " " } else { "" };
                        let modifiers_concatenated = format!("{constness}{glue}{asyncness}");
                        self.dcx().emit_err(errors::PolarityAndModifiers {
                            polarity_span,
                            polarity: polarity.as_str(),
                            modifiers_span: modifier_lo.to(modifier_hi),
                            modifiers_concatenated,
                        });
                    }
                }
            }
        }

        Ok(TraitBoundModifiers { constness, asyncness, polarity })
    }

    /// Parses a type bound according to:
    /// ```ebnf
    /// TY_BOUND = TY_BOUND_NOPAREN | (TY_BOUND_NOPAREN)
    /// TY_BOUND_NOPAREN = [for<GENERIC_PARAMS> CONSTNESS ASYNCNESS | POLARITY] SIMPLE_PATH
    /// ```
    ///
    /// For example, this grammar accepts `for<'a: 'b> ~const ?m::Trait<'a>`.
    fn parse_generic_ty_bound(
        &mut self,
        lo: Span,
        has_parens: bool,
        leading_token: &Token,
    ) -> PResult<'a, GenericBound> {
        let (mut lifetime_defs, binder_span) = self.parse_late_bound_lifetime_defs()?;

        let modifiers_lo = self.token.span;
        let modifiers = self.parse_trait_bound_modifiers()?;
        let modifiers_span = modifiers_lo.to(self.prev_token.span);

        if let Some(binder_span) = binder_span {
            match modifiers.polarity {
                BoundPolarity::Negative(polarity_span) | BoundPolarity::Maybe(polarity_span) => {
                    self.dcx().emit_err(errors::BinderAndPolarity {
                        binder_span,
                        polarity_span,
                        polarity: modifiers.polarity.as_str(),
                    });
                }
                BoundPolarity::Positive => {}
            }
        }

        // Recover erroneous lifetime bound with modifiers or binder.
        // e.g. `T: for<'a> 'a` or `T: ~const 'a`.
        if self.token.is_lifetime() {
            let _: ErrorGuaranteed = self.error_lt_bound_with_modifiers(modifiers, binder_span);
            return self.parse_generic_lt_bound(lo, has_parens);
        }

        if let (more_lifetime_defs, Some(binder_span)) = self.parse_late_bound_lifetime_defs()? {
            lifetime_defs.extend(more_lifetime_defs);
            self.dcx().emit_err(errors::BinderBeforeModifiers { binder_span, modifiers_span });
        }

        let mut path = if self.token.is_keyword(kw::Fn)
            && self.look_ahead(1, |t| *t == TokenKind::OpenParen)
            && let Some(path) = self.recover_path_from_fn()
        {
            path
        } else if !self.token.is_path_start() && self.token.can_begin_type() {
            let ty = self.parse_ty_no_plus()?;
            // Instead of finding a path (a trait), we found a type.
            let mut err = self.dcx().struct_span_err(ty.span, "expected a trait, found type");

            // If we can recover, try to extract a path from the type. Note
            // that we do not use the try operator when parsing the type because
            // if it fails then we get a parser error which we don't want (we're trying
            // to recover from errors, not make more).
            let path = if self.may_recover() {
                let (span, message, sugg, path, applicability) = match &ty.kind {
                    TyKind::Ptr(..) | TyKind::Ref(..)
                        if let TyKind::Path(_, path) = &ty.peel_refs().kind =>
                    {
                        (
                            ty.span.until(path.span),
                            "consider removing the indirection",
                            "",
                            path,
                            Applicability::MaybeIncorrect,
                        )
                    }
                    TyKind::ImplTrait(_, bounds)
                        if let [GenericBound::Trait(tr, ..), ..] = bounds.as_slice() =>
                    {
                        (
                            ty.span.until(tr.span),
                            "use the trait bounds directly",
                            "",
                            &tr.trait_ref.path,
                            Applicability::MachineApplicable,
                        )
                    }
                    _ => return Err(err),
                };

                err.span_suggestion_verbose(span, message, sugg, applicability);

                path.clone()
            } else {
                return Err(err);
            };

            err.emit();

            path
        } else {
            self.parse_path(PathStyle::Type)?
        };

        if self.may_recover() && self.token == TokenKind::OpenParen {
            self.recover_fn_trait_with_lifetime_params(&mut path, &mut lifetime_defs)?;
        }

        if has_parens {
            // Someone has written something like `&dyn (Trait + Other)`. The correct code
            // would be `&(dyn Trait + Other)`
            if self.token.is_like_plus() && leading_token.is_keyword(kw::Dyn) {
                let bounds = vec![];
                self.parse_remaining_bounds(bounds, true)?;
                self.expect(exp!(CloseParen))?;
                self.dcx().emit_err(errors::IncorrectParensTraitBounds {
                    span: vec![lo, self.prev_token.span],
                    sugg: errors::IncorrectParensTraitBoundsSugg {
                        wrong_span: leading_token.span.shrink_to_hi().to(lo),
                        new_span: leading_token.span.shrink_to_lo(),
                    },
                });
            } else {
                self.expect(exp!(CloseParen))?;
            }
        }

        let poly_trait =
            PolyTraitRef::new(lifetime_defs, path, modifiers, lo.to(self.prev_token.span));
        Ok(GenericBound::Trait(poly_trait))
    }

    // recovers a `Fn(..)` parenthesized-style path from `fn(..)`
    fn recover_path_from_fn(&mut self) -> Option<ast::Path> {
        let fn_token_span = self.token.span;
        self.bump();
        let args_lo = self.token.span;
        let snapshot = self.create_snapshot_for_diagnostic();
        match self.parse_fn_decl(|_| false, AllowPlus::No, RecoverReturnSign::OnlyFatArrow) {
            Ok(decl) => {
                self.dcx().emit_err(ExpectedFnPathFoundFnKeyword { fn_token_span });
                Some(ast::Path {
                    span: fn_token_span.to(self.prev_token.span),
                    segments: thin_vec![ast::PathSegment {
                        ident: Ident::new(sym::Fn, fn_token_span),
                        id: DUMMY_NODE_ID,
                        args: Some(P(ast::GenericArgs::Parenthesized(ast::ParenthesizedArgs {
                            span: args_lo.to(self.prev_token.span),
                            inputs: decl.inputs.iter().map(|a| a.ty.clone()).collect(),
                            inputs_span: args_lo.until(decl.output.span()),
                            output: decl.output.clone(),
                        }))),
                    }],
                    tokens: None,
                })
            }
            Err(diag) => {
                diag.cancel();
                self.restore_snapshot(snapshot);
                None
            }
        }
    }

    /// Optionally parses `for<$generic_params>`.
    pub(super) fn parse_late_bound_lifetime_defs(
        &mut self,
    ) -> PResult<'a, (ThinVec<GenericParam>, Option<Span>)> {
        if self.eat_keyword(exp!(For)) {
            let lo = self.token.span;
            self.expect_lt()?;
            let params = self.parse_generic_params()?;
            self.expect_gt()?;
            // We rely on AST validation to rule out invalid cases: There must not be
            // type or const parameters, and parameters must not have bounds.
            Ok((params, Some(lo.to(self.prev_token.span))))
        } else {
            Ok((ThinVec::new(), None))
        }
    }

    /// Recover from `Fn`-family traits (Fn, FnMut, FnOnce) with lifetime arguments
    /// (e.g. `FnOnce<'a>(&'a str) -> bool`). Up to generic arguments have already
    /// been eaten.
    fn recover_fn_trait_with_lifetime_params(
        &mut self,
        fn_path: &mut ast::Path,
        lifetime_defs: &mut ThinVec<GenericParam>,
    ) -> PResult<'a, ()> {
        let fn_path_segment = fn_path.segments.last_mut().unwrap();
        let generic_args = if let Some(p_args) = &fn_path_segment.args {
            p_args.clone().into_inner()
        } else {
            // Normally it wouldn't come here because the upstream should have parsed
            // generic parameters (otherwise it's impossible to call this function).
            return Ok(());
        };
        let lifetimes =
            if let ast::GenericArgs::AngleBracketed(ast::AngleBracketedArgs { span: _, args }) =
                &generic_args
            {
                args.into_iter()
                    .filter_map(|arg| {
                        if let ast::AngleBracketedArg::Arg(generic_arg) = arg
                            && let ast::GenericArg::Lifetime(lifetime) = generic_arg
                        {
                            Some(lifetime)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
        // Only try to recover if the trait has lifetime params.
        if lifetimes.is_empty() {
            return Ok(());
        }

        // Parse `(T, U) -> R`.
        let inputs_lo = self.token.span;
        let inputs: ThinVec<_> =
            self.parse_fn_params(|_| false)?.into_iter().map(|input| input.ty).collect();
        let inputs_span = inputs_lo.to(self.prev_token.span);
        let output = self.parse_ret_ty(AllowPlus::No, RecoverQPath::No, RecoverReturnSign::No)?;
        let args = ast::ParenthesizedArgs {
            span: fn_path_segment.span().to(self.prev_token.span),
            inputs,
            inputs_span,
            output,
        }
        .into();
        *fn_path_segment = ast::PathSegment {
            ident: fn_path_segment.ident,
            args: Some(args),
            id: ast::DUMMY_NODE_ID,
        };

        // Convert parsed `<'a>` in `Fn<'a>` into `for<'a>`.
        let mut generic_params = lifetimes
            .iter()
            .map(|lt| GenericParam {
                id: lt.id,
                ident: lt.ident,
                attrs: ast::AttrVec::new(),
                bounds: Vec::new(),
                is_placeholder: false,
                kind: ast::GenericParamKind::Lifetime,
                colon_span: None,
            })
            .collect::<ThinVec<GenericParam>>();
        lifetime_defs.append(&mut generic_params);

        let generic_args_span = generic_args.span();
        let snippet = format!(
            "for<{}> ",
            lifetimes.iter().map(|lt| lt.ident.as_str()).intersperse(", ").collect::<String>(),
        );
        let before_fn_path = fn_path.span.shrink_to_lo();
        self.dcx()
            .struct_span_err(generic_args_span, "`Fn` traits cannot take lifetime parameters")
            .with_multipart_suggestion(
                "consider using a higher-ranked trait bound instead",
                vec![(generic_args_span, "".to_owned()), (before_fn_path, snippet)],
                Applicability::MaybeIncorrect,
            )
            .emit();
        Ok(())
    }

    pub(super) fn check_lifetime(&mut self) -> bool {
        self.expected_token_types.insert(TokenType::Lifetime);
        self.token.is_lifetime()
    }

    /// Parses a single lifetime `'a` or panics.
    pub(super) fn expect_lifetime(&mut self) -> Lifetime {
        if let Some((ident, is_raw)) = self.token.lifetime() {
            if matches!(is_raw, IdentIsRaw::No)
                && ident.without_first_quote().is_reserved()
                && ![kw::UnderscoreLifetime, kw::StaticLifetime].contains(&ident.name)
            {
                self.dcx().emit_err(errors::KeywordLifetime { span: ident.span });
            }

            self.bump();
            Lifetime { ident, id: ast::DUMMY_NODE_ID }
        } else {
            self.dcx().span_bug(self.token.span, "not a lifetime")
        }
    }

    pub(super) fn mk_ty(&self, span: Span, kind: TyKind) -> P<Ty> {
        P(Ty { kind, span, id: ast::DUMMY_NODE_ID, tokens: None })
    }
}
