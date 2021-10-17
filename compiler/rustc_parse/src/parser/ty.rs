use super::{Parser, PathStyle, TokenType};

use crate::{maybe_recover_from_interpolated_ty_qpath, maybe_whole};

use rustc_ast::ptr::P;
use rustc_ast::token::{self, Token, TokenKind};
use rustc_ast::{self as ast, BareFnTy, FnRetTy, GenericParam, Lifetime, MutTy, Ty, TyKind};
use rustc_ast::{GenericBound, GenericBounds, MacCall, Mutability};
use rustc_ast::{PolyTraitRef, TraitBoundModifier, TraitObjectSyntax};
use rustc_errors::{pluralize, struct_span_err, Applicability, PResult};
use rustc_span::source_map::Span;
use rustc_span::symbol::{kw, sym};

/// Any `?` or `~const` modifiers that appear at the start of a bound.
struct BoundModifiers {
    /// `?Trait`.
    maybe: Option<Span>,

    /// `~const Trait`.
    maybe_const: Option<Span>,
}

impl BoundModifiers {
    fn to_trait_bound_modifier(&self) -> TraitBoundModifier {
        match (self.maybe, self.maybe_const) {
            (None, None) => TraitBoundModifier::None,
            (Some(_), None) => TraitBoundModifier::Maybe,
            (None, Some(_)) => TraitBoundModifier::MaybeConst,
            (Some(_), Some(_)) => TraitBoundModifier::MaybeConstMaybe,
        }
    }
}

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

/// Signals whether parsing a type should recover `->`.
///
/// More specifically, when parsing a function like:
/// ```rust
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
    t == &token::ModSep || t == &token::Lt || t == &token::BinOp(token::Shl)
}

impl<'a> Parser<'a> {
    /// Parses a type.
    pub fn parse_ty(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::Yes,
            AllowCVariadic::No,
            RecoverQPath::Yes,
            RecoverReturnSign::Yes,
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
        )
    }

    /// Parse a type without recovering `:` as `->` to avoid breaking code such as `where fn() : for<'a>`
    pub(super) fn parse_ty_for_where_clause(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(
            AllowPlus::Yes,
            AllowCVariadic::Yes,
            RecoverQPath::Yes,
            RecoverReturnSign::OnlyFatArrow,
        )
    }

    /// Parses an optional return type `[ -> TY ]` in a function declaration.
    pub(super) fn parse_ret_ty(
        &mut self,
        allow_plus: AllowPlus,
        recover_qpath: RecoverQPath,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, FnRetTy> {
        Ok(if self.eat(&token::RArrow) {
            // FIXME(Centril): Can we unconditionally `allow_plus`?
            let ty = self.parse_ty_common(
                allow_plus,
                AllowCVariadic::No,
                recover_qpath,
                recover_return_sign,
            )?;
            FnRetTy::Ty(ty)
        } else if recover_return_sign.can_recover(&self.token.kind) {
            // Don't `eat` to prevent `=>` from being added as an expected token which isn't
            // actually expected and could only confuse users
            self.bump();
            self.struct_span_err(self.prev_token.span, "return types are denoted using `->`")
                .span_suggestion_short(
                    self.prev_token.span,
                    "use `->` instead",
                    "->".to_string(),
                    Applicability::MachineApplicable,
                )
                .emit();
            let ty = self.parse_ty_common(
                allow_plus,
                AllowCVariadic::No,
                recover_qpath,
                recover_return_sign,
            )?;
            FnRetTy::Ty(ty)
        } else {
            FnRetTy::Default(self.token.span.shrink_to_lo())
        })
    }

    fn parse_ty_common(
        &mut self,
        allow_plus: AllowPlus,
        allow_c_variadic: AllowCVariadic,
        recover_qpath: RecoverQPath,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, P<Ty>> {
        let allow_qpath_recovery = recover_qpath == RecoverQPath::Yes;
        maybe_recover_from_interpolated_ty_qpath!(self, allow_qpath_recovery);
        maybe_whole!(self, NtTy, |x| x);

        let lo = self.token.span;
        let mut impl_dyn_multi = false;
        let kind = if self.check(&token::OpenDelim(token::Paren)) {
            self.parse_ty_tuple_or_parens(lo, allow_plus)?
        } else if self.eat(&token::Not) {
            // Never type `!`
            TyKind::Never
        } else if self.eat(&token::BinOp(token::Star)) {
            self.parse_ty_ptr()?
        } else if self.eat(&token::OpenDelim(token::Bracket)) {
            self.parse_array_or_slice_ty()?
        } else if self.check(&token::BinOp(token::And)) || self.check(&token::AndAnd) {
            // Reference
            self.expect_and()?;
            self.parse_borrowed_pointee()?
        } else if self.eat_keyword_noexpect(kw::Typeof) {
            self.parse_typeof_ty()?
        } else if self.eat_keyword(kw::Underscore) {
            // A type to be inferred `_`
            TyKind::Infer
        } else if self.check_fn_front_matter(false) {
            // Function pointer type
            self.parse_ty_bare_fn(lo, Vec::new(), recover_return_sign)?
        } else if self.check_keyword(kw::For) {
            // Function pointer type or bound list (trait object type) starting with a poly-trait.
            //   `for<'lt> [unsafe] [extern "ABI"] fn (&'lt S) -> T`
            //   `for<'lt> Trait1<'lt> + Trait2 + 'a`
            let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
            if self.check_fn_front_matter(false) {
                self.parse_ty_bare_fn(lo, lifetime_defs, recover_return_sign)?
            } else {
                let path = self.parse_path(PathStyle::Type)?;
                let parse_plus = allow_plus == AllowPlus::Yes && self.check_plus();
                self.parse_remaining_bounds_path(lifetime_defs, path, lo, parse_plus)?
            }
        } else if self.eat_keyword(kw::Impl) {
            self.parse_impl_ty(&mut impl_dyn_multi)?
        } else if self.is_explicit_dyn_type() {
            self.parse_dyn_ty(&mut impl_dyn_multi)?
        } else if self.eat_lt() {
            // Qualified path
            let (qself, path) = self.parse_qpath(PathStyle::Type)?;
            TyKind::Path(Some(qself), path)
        } else if self.check_path() {
            self.parse_path_start_ty(lo, allow_plus)?
        } else if self.can_begin_bound() {
            self.parse_bare_trait_object(lo, allow_plus)?
        } else if self.eat(&token::DotDotDot) {
            if allow_c_variadic == AllowCVariadic::Yes {
                TyKind::CVarArgs
            } else {
                // FIXME(Centril): Should we just allow `...` syntactically
                // anywhere in a type and use semantic restrictions instead?
                self.error_illegal_c_varadic_ty(lo);
                TyKind::Err
            }
        } else {
            let msg = format!("expected type, found {}", super::token_descr(&self.token));
            let mut err = self.struct_span_err(self.token.span, &msg);
            err.span_label(self.token.span, "expected type");
            self.maybe_annotate_with_ascription(&mut err, true);
            return Err(err);
        };

        let span = lo.to(self.prev_token.span);
        let ty = self.mk_ty(span, kind);

        // Try to recover from use of `+` with incorrect priority.
        self.maybe_report_ambiguous_plus(allow_plus, impl_dyn_multi, &ty);
        self.maybe_recover_from_bad_type_plus(allow_plus, &ty)?;
        self.maybe_recover_from_bad_qpath(ty, allow_qpath_recovery)
    }

    /// Parses either:
    /// - `(TYPE)`, a parenthesized type.
    /// - `(TYPE,)`, a tuple with a single field of type TYPE.
    fn parse_ty_tuple_or_parens(&mut self, lo: Span, allow_plus: AllowPlus) -> PResult<'a, TyKind> {
        let mut trailing_plus = false;
        let (ts, trailing) = self.parse_paren_comma_seq(|p| {
            let ty = p.parse_ty()?;
            trailing_plus = p.prev_token.kind == TokenKind::BinOp(token::Plus);
            Ok(ty)
        })?;

        if ts.len() == 1 && !trailing {
            let ty = ts.into_iter().next().unwrap().into_inner();
            let maybe_bounds = allow_plus == AllowPlus::Yes && self.token.is_like_plus();
            match ty.kind {
                // `(TY_BOUND_NOPAREN) + BOUND + ...`.
                TyKind::Path(None, path) if maybe_bounds => {
                    self.parse_remaining_bounds_path(Vec::new(), path, lo, true)
                }
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
        let lt_no_plus = self.check_lifetime() && !self.look_ahead(1, |t| t.is_like_plus());
        let bounds = self.parse_generic_bounds_common(allow_plus, None)?;
        if lt_no_plus {
            self.struct_span_err(lo, "lifetime in trait object type must be followed by `+`").emit()
        }
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::None))
    }

    fn parse_remaining_bounds_path(
        &mut self,
        generic_params: Vec<GenericParam>,
        path: ast::Path,
        lo: Span,
        parse_plus: bool,
    ) -> PResult<'a, TyKind> {
        let poly_trait_ref = PolyTraitRef::new(generic_params, path, lo.to(self.prev_token.span));
        let bounds = vec![GenericBound::Trait(poly_trait_ref, TraitBoundModifier::None)];
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
            bounds.append(&mut self.parse_generic_bounds(Some(self.prev_token.span))?);
        }
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::None))
    }

    /// Parses a raw pointer type: `*[const | mut] $type`.
    fn parse_ty_ptr(&mut self) -> PResult<'a, TyKind> {
        let mutbl = self.parse_const_or_mut().unwrap_or_else(|| {
            let span = self.prev_token.span;
            let msg = "expected mut or const in raw pointer type";
            self.struct_span_err(span, msg)
                .span_label(span, msg)
                .help("use `*mut T` or `*const T` as appropriate")
                .emit();
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
            Err(mut err)
                if self.look_ahead(1, |t| t.kind == token::CloseDelim(token::Bracket))
                    | self.look_ahead(1, |t| t.kind == token::Semi) =>
            {
                // Recover from `[LIT; EXPR]` and `[LIT]`
                self.bump();
                err.emit();
                self.mk_ty(self.prev_token.span, TyKind::Err)
            }
            Err(err) => return Err(err),
        };

        let ty = if self.eat(&token::Semi) {
            let mut length = self.parse_anon_const_expr()?;
            if let Err(e) = self.expect(&token::CloseDelim(token::Bracket)) {
                // Try to recover from `X<Y, ...>` when `X::<Y, ...>` works
                self.check_mistyped_turbofish_with_multiple_type_params(e, &mut length.value)?;
                self.expect(&token::CloseDelim(token::Bracket))?;
            }
            TyKind::Array(elt_ty, length)
        } else {
            self.expect(&token::CloseDelim(token::Bracket))?;
            TyKind::Slice(elt_ty)
        };

        Ok(ty)
    }

    fn parse_borrowed_pointee(&mut self) -> PResult<'a, TyKind> {
        let and_span = self.prev_token.span;
        let mut opt_lifetime =
            if self.check_lifetime() { Some(self.expect_lifetime()) } else { None };
        let mut mutbl = self.parse_mutability();
        if self.token.is_lifetime() && mutbl == Mutability::Mut && opt_lifetime.is_none() {
            // A lifetime is invalid here: it would be part of a bare trait bound, which requires
            // it to be followed by a plus, but we disallow plus in the pointee type.
            // So we can handle this case as an error here, and suggest `'a mut`.
            // If there *is* a plus next though, handling the error later provides better suggestions
            // (like adding parentheses)
            if !self.look_ahead(1, |t| t.is_like_plus()) {
                let lifetime_span = self.token.span;
                let span = and_span.to(lifetime_span);

                let mut err = self.struct_span_err(span, "lifetime must precede `mut`");
                if let Ok(lifetime_src) = self.span_to_snippet(lifetime_span) {
                    err.span_suggestion(
                        span,
                        "place the lifetime before `mut`",
                        format!("&{} mut", lifetime_src),
                        Applicability::MaybeIncorrect,
                    );
                }
                err.emit();

                opt_lifetime = Some(self.expect_lifetime());
            }
        } else if self.token.is_keyword(kw::Dyn)
            && mutbl == Mutability::Not
            && self.look_ahead(1, |t| t.is_keyword(kw::Mut))
        {
            // We have `&dyn mut ...`, which is invalid and should be `&mut dyn ...`.
            let span = and_span.to(self.look_ahead(1, |t| t.span));
            let mut err = self.struct_span_err(span, "`mut` must precede `dyn`");
            err.span_suggestion(
                span,
                "place `mut` before `dyn`",
                "&mut dyn".to_string(),
                Applicability::MachineApplicable,
            );
            err.emit();

            // Recovery
            mutbl = Mutability::Mut;
            let (dyn_tok, dyn_tok_sp) = (self.token.clone(), self.token_spacing);
            self.bump();
            self.bump_with((dyn_tok, dyn_tok_sp));
        }
        let ty = self.parse_ty_no_plus()?;
        Ok(TyKind::Rptr(opt_lifetime, MutTy { ty, mutbl }))
    }

    // Parses the `typeof(EXPR)`.
    // To avoid ambiguity, the type is surrounded by parentheses.
    fn parse_typeof_ty(&mut self) -> PResult<'a, TyKind> {
        self.expect(&token::OpenDelim(token::Paren))?;
        let expr = self.parse_anon_const_expr()?;
        self.expect(&token::CloseDelim(token::Paren))?;
        Ok(TyKind::Typeof(expr))
    }

    /// Parses a function pointer type (`TyKind::BareFn`).
    /// ```
    /// [unsafe] [extern "ABI"] fn (S) -> T
    ///  ^~~~~^          ^~~~^     ^~^    ^
    ///    |               |        |     |
    ///    |               |        |   Return type
    /// Function Style    ABI  Parameter types
    /// ```
    /// We actually parse `FnHeader FnDecl`, but we error on `const` and `async` qualifiers.
    fn parse_ty_bare_fn(
        &mut self,
        lo: Span,
        params: Vec<GenericParam>,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, TyKind> {
        let ast::FnHeader { ext, unsafety, constness, asyncness } = self.parse_fn_front_matter()?;
        let decl = self.parse_fn_decl(|_| false, AllowPlus::No, recover_return_sign)?;
        let whole_span = lo.to(self.prev_token.span);
        if let ast::Const::Yes(span) = constness {
            self.error_fn_ptr_bad_qualifier(whole_span, span, "const");
        }
        if let ast::Async::Yes { span, .. } = asyncness {
            self.error_fn_ptr_bad_qualifier(whole_span, span, "async");
        }
        Ok(TyKind::BareFn(P(BareFnTy { ext, unsafety, generic_params: params, decl })))
    }

    /// Emit an error for the given bad function pointer qualifier.
    fn error_fn_ptr_bad_qualifier(&self, span: Span, qual_span: Span, qual: &str) {
        self.struct_span_err(span, &format!("an `fn` pointer type cannot be `{}`", qual))
            .span_label(qual_span, format!("`{}` because of this", qual))
            .span_suggestion_short(
                qual_span,
                &format!("remove the `{}` qualifier", qual),
                String::new(),
                Applicability::MaybeIncorrect,
            )
            .emit();
    }

    /// Parses an `impl B0 + ... + Bn` type.
    fn parse_impl_ty(&mut self, impl_dyn_multi: &mut bool) -> PResult<'a, TyKind> {
        // Always parse bounds greedily for better error recovery.
        let bounds = self.parse_generic_bounds(None)?;
        *impl_dyn_multi = bounds.len() > 1 || self.prev_token.kind == TokenKind::BinOp(token::Plus);
        Ok(TyKind::ImplTrait(ast::DUMMY_NODE_ID, bounds))
    }

    /// Is a `dyn B0 + ... + Bn` type allowed here?
    fn is_explicit_dyn_type(&mut self) -> bool {
        self.check_keyword(kw::Dyn)
            && (!self.token.uninterpolated_span().rust_2015()
                || self.look_ahead(1, |t| {
                    t.can_begin_bound() && !can_continue_type_after_non_fn_ident(t)
                }))
    }

    /// Parses a `dyn B0 + ... + Bn` type.
    ///
    /// Note that this does *not* parse bare trait objects.
    fn parse_dyn_ty(&mut self, impl_dyn_multi: &mut bool) -> PResult<'a, TyKind> {
        self.bump(); // `dyn`
        // Always parse bounds greedily for better error recovery.
        let bounds = self.parse_generic_bounds(None)?;
        *impl_dyn_multi = bounds.len() > 1 || self.prev_token.kind == TokenKind::BinOp(token::Plus);
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::Dyn))
    }

    /// Parses a type starting with a path.
    ///
    /// This can be:
    /// 1. a type macro, `mac!(...)`,
    /// 2. a bare trait object, `B0 + ... + Bn`,
    /// 3. or a path, `path::to::MyType`.
    fn parse_path_start_ty(&mut self, lo: Span, allow_plus: AllowPlus) -> PResult<'a, TyKind> {
        // Simple path
        let path = self.parse_path(PathStyle::Type)?;
        if self.eat(&token::Not) {
            // Macro invocation in type position
            Ok(TyKind::MacCall(MacCall {
                path,
                args: self.parse_mac_args()?,
                prior_type_ascription: self.last_type_ascription,
            }))
        } else if allow_plus == AllowPlus::Yes && self.check_plus() {
            // `Trait1 + Trait2 + 'a`
            self.parse_remaining_bounds_path(Vec::new(), path, lo, true)
        } else {
            // Just a type path.
            Ok(TyKind::Path(None, path))
        }
    }

    fn error_illegal_c_varadic_ty(&self, lo: Span) {
        struct_span_err!(
            self.sess.span_diagnostic,
            lo.to(self.prev_token.span),
            E0743,
            "C-variadic type `...` may not be nested inside another type",
        )
        .emit();
    }

    pub(super) fn parse_generic_bounds(
        &mut self,
        colon_span: Option<Span>,
    ) -> PResult<'a, GenericBounds> {
        self.parse_generic_bounds_common(AllowPlus::Yes, colon_span)
    }

    /// Parses bounds of a type parameter `BOUND + BOUND + ...`, possibly with trailing `+`.
    ///
    /// See `parse_generic_bound` for the `BOUND` grammar.
    fn parse_generic_bounds_common(
        &mut self,
        allow_plus: AllowPlus,
        colon_span: Option<Span>,
    ) -> PResult<'a, GenericBounds> {
        let mut bounds = Vec::new();
        let mut negative_bounds = Vec::new();

        while self.can_begin_bound() || self.token.is_keyword(kw::Dyn) {
            if self.token.is_keyword(kw::Dyn) {
                // Account for `&dyn Trait + dyn Other`.
                self.struct_span_err(self.token.span, "invalid `dyn` keyword")
                    .help("`dyn` is only needed at the start of a trait `+`-separated list")
                    .span_suggestion(
                        self.token.span,
                        "remove this keyword",
                        String::new(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
                self.bump();
            }
            match self.parse_generic_bound()? {
                Ok(bound) => bounds.push(bound),
                Err(neg_sp) => negative_bounds.push(neg_sp),
            }
            if allow_plus == AllowPlus::No || !self.eat_plus() {
                break;
            }
        }

        if !negative_bounds.is_empty() {
            self.error_negative_bounds(colon_span, &bounds, negative_bounds);
        }

        Ok(bounds)
    }

    /// Can the current token begin a bound?
    fn can_begin_bound(&mut self) -> bool {
        // This needs to be synchronized with `TokenKind::can_begin_bound`.
        self.check_path()
        || self.check_lifetime()
        || self.check(&token::Not) // Used for error reporting only.
        || self.check(&token::Question)
        || self.check(&token::Tilde)
        || self.check_keyword(kw::For)
        || self.check(&token::OpenDelim(token::Paren))
    }

    fn error_negative_bounds(
        &self,
        colon_span: Option<Span>,
        bounds: &[GenericBound],
        negative_bounds: Vec<Span>,
    ) {
        let negative_bounds_len = negative_bounds.len();
        let last_span = *negative_bounds.last().expect("no negative bounds, but still error?");
        let mut err = self.struct_span_err(negative_bounds, "negative bounds are not supported");
        err.span_label(last_span, "negative bounds are not supported");
        if let Some(bound_list) = colon_span {
            let bound_list = bound_list.to(self.prev_token.span);
            let mut new_bound_list = String::new();
            if !bounds.is_empty() {
                let mut snippets = bounds.iter().map(|bound| self.span_to_snippet(bound.span()));
                while let Some(Ok(snippet)) = snippets.next() {
                    new_bound_list.push_str(" + ");
                    new_bound_list.push_str(&snippet);
                }
                new_bound_list = new_bound_list.replacen(" +", ":", 1);
            }
            err.tool_only_span_suggestion(
                bound_list,
                &format!("remove the bound{}", pluralize!(negative_bounds_len)),
                new_bound_list,
                Applicability::MachineApplicable,
            );
        }
        err.emit();
    }

    /// Parses a bound according to the grammar:
    /// ```
    /// BOUND = TY_BOUND | LT_BOUND
    /// ```
    fn parse_generic_bound(&mut self) -> PResult<'a, Result<GenericBound, Span>> {
        let anchor_lo = self.prev_token.span;
        let lo = self.token.span;
        let has_parens = self.eat(&token::OpenDelim(token::Paren));
        let inner_lo = self.token.span;
        let is_negative = self.eat(&token::Not);

        let modifiers = self.parse_ty_bound_modifiers()?;
        let bound = if self.token.is_lifetime() {
            self.error_lt_bound_with_modifiers(modifiers);
            self.parse_generic_lt_bound(lo, inner_lo, has_parens)?
        } else {
            self.parse_generic_ty_bound(lo, has_parens, modifiers)?
        };

        Ok(if is_negative { Err(anchor_lo.to(self.prev_token.span)) } else { Ok(bound) })
    }

    /// Parses a lifetime ("outlives") bound, e.g. `'a`, according to:
    /// ```
    /// LT_BOUND = LIFETIME
    /// ```
    fn parse_generic_lt_bound(
        &mut self,
        lo: Span,
        inner_lo: Span,
        has_parens: bool,
    ) -> PResult<'a, GenericBound> {
        let bound = GenericBound::Outlives(self.expect_lifetime());
        if has_parens {
            // FIXME(Centril): Consider not erroring here and accepting `('lt)` instead,
            // possibly introducing `GenericBound::Paren(P<GenericBound>)`?
            self.recover_paren_lifetime(lo, inner_lo)?;
        }
        Ok(bound)
    }

    /// Emits an error if any trait bound modifiers were present.
    fn error_lt_bound_with_modifiers(&self, modifiers: BoundModifiers) {
        if let Some(span) = modifiers.maybe_const {
            self.struct_span_err(
                span,
                "`~const` may only modify trait bounds, not lifetime bounds",
            )
            .emit();
        }

        if let Some(span) = modifiers.maybe {
            self.struct_span_err(span, "`?` may only modify trait bounds, not lifetime bounds")
                .emit();
        }
    }

    /// Recover on `('lifetime)` with `(` already eaten.
    fn recover_paren_lifetime(&mut self, lo: Span, inner_lo: Span) -> PResult<'a, ()> {
        let inner_span = inner_lo.to(self.prev_token.span);
        self.expect(&token::CloseDelim(token::Paren))?;
        let mut err = self.struct_span_err(
            lo.to(self.prev_token.span),
            "parenthesized lifetime bounds are not supported",
        );
        if let Ok(snippet) = self.span_to_snippet(inner_span) {
            err.span_suggestion_short(
                lo.to(self.prev_token.span),
                "remove the parentheses",
                snippet,
                Applicability::MachineApplicable,
            );
        }
        err.emit();
        Ok(())
    }

    /// Parses the modifiers that may precede a trait in a bound, e.g. `?Trait` or `~const Trait`.
    ///
    /// If no modifiers are present, this does not consume any tokens.
    ///
    /// ```
    /// TY_BOUND_MODIFIERS = ["~const"] ["?"]
    /// ```
    fn parse_ty_bound_modifiers(&mut self) -> PResult<'a, BoundModifiers> {
        let maybe_const = if self.eat(&token::Tilde) {
            let tilde = self.prev_token.span;
            self.expect_keyword(kw::Const)?;
            let span = tilde.to(self.prev_token.span);
            self.sess.gated_spans.gate(sym::const_trait_impl, span);
            Some(span)
        } else {
            None
        };

        let maybe = if self.eat(&token::Question) { Some(self.prev_token.span) } else { None };

        Ok(BoundModifiers { maybe, maybe_const })
    }

    /// Parses a type bound according to:
    /// ```
    /// TY_BOUND = TY_BOUND_NOPAREN | (TY_BOUND_NOPAREN)
    /// TY_BOUND_NOPAREN = [TY_BOUND_MODIFIERS] [for<LT_PARAM_DEFS>] SIMPLE_PATH
    /// ```
    ///
    /// For example, this grammar accepts `~const ?for<'a: 'b> m::Trait<'a>`.
    fn parse_generic_ty_bound(
        &mut self,
        lo: Span,
        has_parens: bool,
        modifiers: BoundModifiers,
    ) -> PResult<'a, GenericBound> {
        let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
        let path = self.parse_path(PathStyle::Type)?;
        if has_parens {
            if self.token.is_like_plus() {
                // Someone has written something like `&dyn (Trait + Other)`. The correct code
                // would be `&(dyn Trait + Other)`, but we don't have access to the appropriate
                // span to suggest that. When written as `&dyn Trait + Other`, an appropriate
                // suggestion is given.
                let bounds = vec![];
                self.parse_remaining_bounds(bounds, true)?;
                self.expect(&token::CloseDelim(token::Paren))?;
                let sp = vec![lo, self.prev_token.span];
                let sugg: Vec<_> = sp.iter().map(|sp| (*sp, String::new())).collect();
                self.struct_span_err(sp, "incorrect braces around trait bounds")
                    .multipart_suggestion(
                        "remove the parentheses",
                        sugg,
                        Applicability::MachineApplicable,
                    )
                    .emit();
            } else {
                self.expect(&token::CloseDelim(token::Paren))?;
            }
        }

        let modifier = modifiers.to_trait_bound_modifier();
        let poly_trait = PolyTraitRef::new(lifetime_defs, path, lo.to(self.prev_token.span));
        Ok(GenericBound::Trait(poly_trait, modifier))
    }

    /// Optionally parses `for<$generic_params>`.
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

    pub(super) fn check_lifetime(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Lifetime);
        self.token.is_lifetime()
    }

    /// Parses a single lifetime `'a` or panics.
    pub(super) fn expect_lifetime(&mut self) -> Lifetime {
        if let Some(ident) = self.token.lifetime() {
            self.bump();
            Lifetime { ident, id: ast::DUMMY_NODE_ID }
        } else {
            self.span_bug(self.token.span, "not a lifetime")
        }
    }

    pub(super) fn mk_ty(&self, span: Span, kind: TyKind) -> P<Ty> {
        P(Ty { kind, span, id: ast::DUMMY_NODE_ID, tokens: None })
    }
}
