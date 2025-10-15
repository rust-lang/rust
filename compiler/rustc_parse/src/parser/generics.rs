use rustc_ast::{
    self as ast, AttrVec, DUMMY_NODE_ID, GenericBounds, GenericParam, GenericParamKind, TyKind,
    WhereClause, token,
};
use rustc_errors::{Applicability, PResult};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::ThinVec;

use super::{ForceCollect, Parser, Trailing, UsePreAttrPos};
use crate::errors::{
    self, MultipleWhereClauses, UnexpectedDefaultValueForLifetimeInGenericParameters,
    UnexpectedSelfInGenericParameters, WhereClauseBeforeTupleStructBody,
    WhereClauseBeforeTupleStructBodySugg,
};
use crate::exp;

enum PredicateKindOrStructBody {
    PredicateKind(ast::WherePredicateKind),
    StructBody(ThinVec<ast::FieldDef>),
}

impl<'a> Parser<'a> {
    /// Parses bounds of a lifetime parameter `BOUND + BOUND + BOUND`, possibly with trailing `+`.
    ///
    /// ```text
    /// BOUND = LT_BOUND (e.g., `'a`)
    /// ```
    fn parse_lt_param_bounds(&mut self) -> GenericBounds {
        let mut lifetimes = Vec::new();
        while self.check_lifetime() {
            lifetimes.push(ast::GenericBound::Outlives(self.expect_lifetime()));

            if !self.eat_plus() {
                break;
            }
        }
        lifetimes
    }

    /// Matches `typaram = IDENT (`?` unbound)? optbounds ( EQ ty )?`.
    fn parse_ty_param(&mut self, preceding_attrs: AttrVec) -> PResult<'a, GenericParam> {
        let ident = self.parse_ident()?;

        // We might have a typo'd `Const` that was parsed as a type parameter.
        if self.may_recover()
            && ident.name.as_str().to_ascii_lowercase() == kw::Const.as_str()
            && self.check_ident()
        // `Const` followed by IDENT
        {
            return self.recover_const_param_with_mistyped_const(preceding_attrs, ident);
        }

        // Parse optional colon and param bounds.
        let mut colon_span = None;
        let bounds = if self.eat(exp!(Colon)) {
            colon_span = Some(self.prev_token.span);
            // recover from `impl Trait` in type param bound
            if self.token.is_keyword(kw::Impl) {
                let impl_span = self.token.span;
                let snapshot = self.create_snapshot_for_diagnostic();
                match self.parse_ty() {
                    Ok(p) => {
                        if let TyKind::ImplTrait(_, bounds) = &p.kind {
                            let span = impl_span.to(self.token.span.shrink_to_lo());
                            let mut err = self.dcx().struct_span_err(
                                span,
                                "expected trait bound, found `impl Trait` type",
                            );
                            err.span_label(span, "not a trait");
                            if let [bound, ..] = &bounds[..] {
                                err.span_suggestion_verbose(
                                    impl_span.until(bound.span()),
                                    "use the trait bounds directly",
                                    String::new(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            return Err(err);
                        }
                    }
                    Err(err) => {
                        err.cancel();
                    }
                }
                self.restore_snapshot(snapshot);
            }
            self.parse_generic_bounds()?
        } else {
            Vec::new()
        };

        let default = if self.eat(exp!(Eq)) { Some(self.parse_ty()?) } else { None };
        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs,
            bounds,
            kind: GenericParamKind::Type { default },
            is_placeholder: false,
            colon_span,
        })
    }

    pub(crate) fn parse_const_param(
        &mut self,
        preceding_attrs: AttrVec,
    ) -> PResult<'a, GenericParam> {
        let const_span = self.token.span;

        self.expect_keyword(exp!(Const))?;
        let ident = self.parse_ident()?;
        self.expect(exp!(Colon))?;
        let ty = self.parse_ty()?;

        // Parse optional const generics default value.
        let default = if self.eat(exp!(Eq)) { Some(self.parse_const_arg()?) } else { None };
        let span = if let Some(ref default) = default {
            const_span.to(default.value.span)
        } else {
            const_span.to(ty.span)
        };

        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs,
            bounds: Vec::new(),
            kind: GenericParamKind::Const { ty, span, default },
            is_placeholder: false,
            colon_span: None,
        })
    }

    pub(crate) fn recover_const_param_with_mistyped_const(
        &mut self,
        preceding_attrs: AttrVec,
        mistyped_const_ident: Ident,
    ) -> PResult<'a, GenericParam> {
        let ident = self.parse_ident()?;
        self.expect(exp!(Colon))?;
        let ty = self.parse_ty()?;

        // Parse optional const generics default value.
        let default = if self.eat(exp!(Eq)) { Some(self.parse_const_arg()?) } else { None };
        let span = if let Some(ref default) = default {
            mistyped_const_ident.span.to(default.value.span)
        } else {
            mistyped_const_ident.span.to(ty.span)
        };

        self.dcx()
            .struct_span_err(
                mistyped_const_ident.span,
                format!("`const` keyword was mistyped as `{}`", mistyped_const_ident.as_str()),
            )
            .with_span_suggestion_verbose(
                mistyped_const_ident.span,
                "use the `const` keyword",
                kw::Const,
                Applicability::MachineApplicable,
            )
            .emit();

        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs,
            bounds: Vec::new(),
            kind: GenericParamKind::Const { ty, span, default },
            is_placeholder: false,
            colon_span: None,
        })
    }

    /// Parse a (possibly empty) list of generic (lifetime, type, const) parameters.
    ///
    /// ```ebnf
    /// GenericParams = (GenericParam ("," GenericParam)* ","?)?
    /// ```
    pub(super) fn parse_generic_params(&mut self) -> PResult<'a, ThinVec<ast::GenericParam>> {
        let mut params = ThinVec::new();
        let mut done = false;
        while !done {
            let attrs = self.parse_outer_attributes()?;
            let param = self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
                if this.eat_keyword_noexpect(kw::SelfUpper) {
                    // `Self` as a generic param is invalid. Here we emit the diagnostic and continue parsing
                    // as if `Self` never existed.
                    this.dcx()
                        .emit_err(UnexpectedSelfInGenericParameters { span: this.prev_token.span });

                    // Eat a trailing comma, if it exists.
                    let _ = this.eat(exp!(Comma));
                }

                let param = if this.check_lifetime() {
                    let lifetime = this.expect_lifetime();
                    // Parse lifetime parameter.
                    let (colon_span, bounds) = if this.eat(exp!(Colon)) {
                        (Some(this.prev_token.span), this.parse_lt_param_bounds())
                    } else {
                        (None, Vec::new())
                    };

                    if this.check_noexpect(&token::Eq) && this.look_ahead(1, |t| t.is_lifetime()) {
                        let lo = this.token.span;
                        // Parse `= 'lifetime`.
                        this.bump(); // `=`
                        this.bump(); // `'lifetime`
                        let span = lo.to(this.prev_token.span);
                        this.dcx().emit_err(UnexpectedDefaultValueForLifetimeInGenericParameters {
                            span,
                        });
                    }

                    Some(ast::GenericParam {
                        ident: lifetime.ident,
                        id: lifetime.id,
                        attrs,
                        bounds,
                        kind: ast::GenericParamKind::Lifetime,
                        is_placeholder: false,
                        colon_span,
                    })
                } else if this.check_keyword(exp!(Const)) {
                    // Parse const parameter.
                    Some(this.parse_const_param(attrs)?)
                } else if this.check_ident() {
                    // Parse type parameter.
                    Some(this.parse_ty_param(attrs)?)
                } else if this.token.can_begin_type() {
                    // Trying to write an associated type bound? (#26271)
                    let snapshot = this.create_snapshot_for_diagnostic();
                    let lo = this.token.span;
                    match this.parse_ty_where_predicate_kind() {
                        Ok(_) => {
                            this.dcx().emit_err(errors::BadAssocTypeBounds {
                                span: lo.to(this.prev_token.span),
                            });
                            // FIXME - try to continue parsing other generics?
                        }
                        Err(err) => {
                            err.cancel();
                            // FIXME - maybe we should overwrite 'self' outside of `collect_tokens`?
                            this.restore_snapshot(snapshot);
                        }
                    }
                    return Ok((None, Trailing::No, UsePreAttrPos::No));
                } else {
                    // Check for trailing attributes and stop parsing.
                    if !attrs.is_empty() {
                        if !params.is_empty() {
                            this.dcx().emit_err(errors::AttrAfterGeneric { span: attrs[0].span });
                        } else {
                            this.dcx()
                                .emit_err(errors::AttrWithoutGenerics { span: attrs[0].span });
                        }
                    }
                    return Ok((None, Trailing::No, UsePreAttrPos::No));
                };

                if !this.eat(exp!(Comma)) {
                    done = true;
                }
                // We just ate the comma, so no need to capture the trailing token.
                Ok((param, Trailing::No, UsePreAttrPos::No))
            })?;

            if let Some(param) = param {
                params.push(param);
            } else {
                break;
            }
        }
        Ok(params)
    }

    /// Parses a set of optional generic type parameter declarations. Where
    /// clauses are not parsed here, and must be added later via
    /// `parse_where_clause()`.
    ///
    /// matches generics = ( ) | ( < > ) | ( < typaramseq ( , )? > ) | ( < lifetimes ( , )? > )
    ///                  | ( < lifetimes , typaramseq ( , )? > )
    /// where   typaramseq = ( typaram ) | ( typaram , typaramseq )
    pub(super) fn parse_generics(&mut self) -> PResult<'a, ast::Generics> {
        // invalid path separator `::` in function definition
        // for example `fn invalid_path_separator::<T>() {}`
        if self.eat_noexpect(&token::PathSep) {
            self.dcx()
                .emit_err(errors::InvalidPathSepInFnDefinition { span: self.prev_token.span });
        }

        let span_lo = self.token.span;
        let (params, span) = if self.eat_lt() {
            let params = self.parse_generic_params()?;
            self.expect_gt_or_maybe_suggest_closing_generics(&params)?;
            (params, span_lo.to(self.prev_token.span))
        } else {
            (ThinVec::new(), self.prev_token.span.shrink_to_hi())
        };
        Ok(ast::Generics {
            params,
            where_clause: WhereClause {
                has_where_token: false,
                predicates: ThinVec::new(),
                span: self.prev_token.span.shrink_to_hi(),
            },
            span,
        })
    }

    /// Parses an experimental fn contract
    /// (`contract_requires(WWW) contract_ensures(ZZZ)`)
    pub(super) fn parse_contract(&mut self) -> PResult<'a, Option<Box<ast::FnContract>>> {
        let requires = if self.eat_keyword_noexpect(exp!(ContractRequires).kw) {
            self.psess.gated_spans.gate(sym::contracts_internals, self.prev_token.span);
            let precond = self.parse_expr()?;
            Some(precond)
        } else {
            None
        };
        let ensures = if self.eat_keyword_noexpect(exp!(ContractEnsures).kw) {
            self.psess.gated_spans.gate(sym::contracts_internals, self.prev_token.span);
            let postcond = self.parse_expr()?;
            Some(postcond)
        } else {
            None
        };
        if requires.is_none() && ensures.is_none() {
            Ok(None)
        } else {
            Ok(Some(Box::new(ast::FnContract { requires, ensures })))
        }
    }

    /// Parses an optional where-clause.
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// where T : Trait<U, V> + 'b, 'a : 'b
    /// ```
    pub(super) fn parse_where_clause(&mut self) -> PResult<'a, WhereClause> {
        self.parse_where_clause_common(None).map(|(clause, _)| clause)
    }

    pub(super) fn parse_struct_where_clause(
        &mut self,
        struct_name: Ident,
        body_insertion_point: Span,
    ) -> PResult<'a, (WhereClause, Option<ThinVec<ast::FieldDef>>)> {
        self.parse_where_clause_common(Some((struct_name, body_insertion_point)))
    }

    fn parse_where_clause_common(
        &mut self,
        struct_: Option<(Ident, Span)>,
    ) -> PResult<'a, (WhereClause, Option<ThinVec<ast::FieldDef>>)> {
        let mut where_clause = WhereClause {
            has_where_token: false,
            predicates: ThinVec::new(),
            span: self.prev_token.span.shrink_to_hi(),
        };
        let mut tuple_struct_body = None;

        if !self.eat_keyword(exp!(Where)) {
            return Ok((where_clause, None));
        }

        if self.eat_noexpect(&token::Colon) {
            let colon_span = self.prev_token.span;
            self.dcx()
                .struct_span_err(colon_span, "unexpected colon after `where`")
                .with_span_suggestion_short(
                    colon_span,
                    "remove the colon",
                    "",
                    Applicability::MachineApplicable,
                )
                .emit();
        }

        where_clause.has_where_token = true;
        let where_lo = self.prev_token.span;

        // We are considering adding generics to the `where` keyword as an alternative higher-rank
        // parameter syntax (as in `where<'a>` or `where<T>`. To avoid that being a breaking
        // change we parse those generics now, but report an error.
        if self.choose_generics_over_qpath(0) {
            let generics = self.parse_generics()?;
            self.dcx().emit_err(errors::WhereOnGenerics { span: generics.span });
        }

        loop {
            let where_sp = where_lo.to(self.prev_token.span);
            let attrs = self.parse_outer_attributes()?;
            let pred_lo = self.token.span;
            let predicate = self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
                for attr in &attrs {
                    self.psess.gated_spans.gate(sym::where_clause_attrs, attr.span);
                }
                let kind = if this.check_lifetime() && this.look_ahead(1, |t| !t.is_like_plus()) {
                    let lifetime = this.expect_lifetime();
                    // Bounds starting with a colon are mandatory, but possibly empty.
                    this.expect(exp!(Colon))?;
                    let bounds = this.parse_lt_param_bounds();
                    Some(ast::WherePredicateKind::RegionPredicate(ast::WhereRegionPredicate {
                        lifetime,
                        bounds,
                    }))
                } else if this.check_type() {
                    match this.parse_ty_where_predicate_kind_or_recover_tuple_struct_body(
                        struct_, pred_lo, where_sp,
                    )? {
                        PredicateKindOrStructBody::PredicateKind(kind) => Some(kind),
                        PredicateKindOrStructBody::StructBody(body) => {
                            tuple_struct_body = Some(body);
                            None
                        }
                    }
                } else {
                    None
                };
                let predicate = kind.map(|kind| ast::WherePredicate {
                    attrs,
                    kind,
                    id: DUMMY_NODE_ID,
                    span: pred_lo.to(this.prev_token.span),
                    is_placeholder: false,
                });
                Ok((predicate, Trailing::No, UsePreAttrPos::No))
            })?;
            match predicate {
                Some(predicate) => where_clause.predicates.push(predicate),
                None => break,
            }

            let prev_token = self.prev_token.span;
            let ate_comma = self.eat(exp!(Comma));

            if self.eat_keyword_noexpect(kw::Where) {
                self.dcx().emit_err(MultipleWhereClauses {
                    span: self.token.span,
                    previous: pred_lo,
                    between: prev_token.shrink_to_hi().to(self.prev_token.span),
                });
            } else if !ate_comma {
                break;
            }
        }

        where_clause.span = where_lo.to(self.prev_token.span);
        Ok((where_clause, tuple_struct_body))
    }

    fn parse_ty_where_predicate_kind_or_recover_tuple_struct_body(
        &mut self,
        struct_: Option<(Ident, Span)>,
        pred_lo: Span,
        where_sp: Span,
    ) -> PResult<'a, PredicateKindOrStructBody> {
        let mut snapshot = None;

        if let Some(struct_) = struct_
            && self.may_recover()
            && self.token == token::OpenParen
        {
            snapshot = Some((struct_, self.create_snapshot_for_diagnostic()));
        };

        match self.parse_ty_where_predicate_kind() {
            Ok(pred) => Ok(PredicateKindOrStructBody::PredicateKind(pred)),
            Err(type_err) => {
                let Some(((struct_name, body_insertion_point), mut snapshot)) = snapshot else {
                    return Err(type_err);
                };

                // Check if we might have encountered an out of place tuple struct body.
                match snapshot.parse_tuple_struct_body() {
                    // Since we don't know the exact reason why we failed to parse the
                    // predicate (we might have stumbled upon something bogus like `(T): ?`),
                    // employ a simple heuristic to weed out some pathological cases:
                    // Look for a semicolon (strong indicator) or anything that might mark
                    // the end of the item (weak indicator) following the body.
                    Ok(body)
                        if matches!(snapshot.token.kind, token::Semi | token::Eof)
                            || snapshot.token.can_begin_item() =>
                    {
                        type_err.cancel();

                        let body_sp = pred_lo.to(snapshot.prev_token.span);
                        let map = self.psess.source_map();

                        self.dcx().emit_err(WhereClauseBeforeTupleStructBody {
                            span: where_sp,
                            name: struct_name.span,
                            body: body_sp,
                            sugg: map.span_to_snippet(body_sp).ok().map(|body| {
                                WhereClauseBeforeTupleStructBodySugg {
                                    left: body_insertion_point.shrink_to_hi(),
                                    snippet: body,
                                    right: map.end_point(where_sp).to(body_sp),
                                }
                            }),
                        });

                        self.restore_snapshot(snapshot);
                        Ok(PredicateKindOrStructBody::StructBody(body))
                    }
                    Ok(_) => Err(type_err),
                    Err(body_err) => {
                        body_err.cancel();
                        Err(type_err)
                    }
                }
            }
        }
    }

    fn parse_ty_where_predicate_kind(&mut self) -> PResult<'a, ast::WherePredicateKind> {
        // Parse optional `for<'a, 'b>`.
        // This `for` is parsed greedily and applies to the whole predicate,
        // the bounded type can have its own `for` applying only to it.
        // Examples:
        // * `for<'a> Trait1<'a>: Trait2<'a /* ok */>`
        // * `(for<'a> Trait1<'a>): Trait2<'a /* not ok */>`
        // * `for<'a> for<'b> Trait1<'a, 'b>: Trait2<'a /* ok */, 'b /* not ok */>`
        let (bound_vars, _) = self.parse_higher_ranked_binder()?;

        // Parse type with mandatory colon and (possibly empty) bounds,
        // or with mandatory equality sign and the second type.
        let ty = self.parse_ty_for_where_clause()?;
        if self.eat(exp!(Colon)) {
            let bounds = self.parse_generic_bounds()?;
            Ok(ast::WherePredicateKind::BoundPredicate(ast::WhereBoundPredicate {
                bound_generic_params: bound_vars,
                bounded_ty: ty,
                bounds,
            }))
        // FIXME: Decide what should be used here, `=` or `==`.
        // FIXME: We are just dropping the binders in lifetime_defs on the floor here.
        } else if self.eat(exp!(Eq)) || self.eat(exp!(EqEq)) {
            let rhs_ty = self.parse_ty()?;
            Ok(ast::WherePredicateKind::EqPredicate(ast::WhereEqPredicate { lhs_ty: ty, rhs_ty }))
        } else {
            self.maybe_recover_bounds_doubled_colon(&ty)?;
            self.unexpected_any()
        }
    }

    pub(super) fn choose_generics_over_qpath(&self, start: usize) -> bool {
        // There's an ambiguity between generic parameters and qualified paths in impls.
        // If we see `<` it may start both, so we have to inspect some following tokens.
        // The following combinations can only start generics,
        // but not qualified paths (with one exception):
        //     `<` `>` - empty generic parameters
        //     `<` `#` - generic parameters with attributes
        //     `<` (LIFETIME|IDENT) `>` - single generic parameter
        //     `<` (LIFETIME|IDENT) `,` - first generic parameter in a list
        //     `<` (LIFETIME|IDENT) `:` - generic parameter with bounds
        //     `<` (LIFETIME|IDENT) `=` - generic parameter with a default
        //     `<` const                - generic const parameter
        //     `<` IDENT `?`            - RECOVERY for `impl<T ?Bound` missing a `:`, meant to
        //                                avoid the `T?` to `Option<T>` recovery for types.
        // The only truly ambiguous case is
        //     `<` IDENT `>` `::` IDENT ...
        // we disambiguate it in favor of generics (`impl<T> ::absolute::Path<T> { ... }`)
        // because this is what almost always expected in practice, qualified paths in impls
        // (`impl <Type>::AssocTy { ... }`) aren't even allowed by type checker at the moment.
        self.look_ahead(start, |t| t == &token::Lt)
            && (self.look_ahead(start + 1, |t| t == &token::Pound || t == &token::Gt)
                || self.look_ahead(start + 1, |t| t.is_lifetime() || t.is_ident())
                    && self.look_ahead(start + 2, |t| {
                        matches!(t.kind, token::Gt | token::Comma | token::Colon | token::Eq)
                        // Recovery-only branch -- this could be removed,
                        // since it only affects diagnostics currently.
                            || t.kind == token::Question
                    })
                || self.is_keyword_ahead(start + 1, &[kw::Const]))
    }
}
