use super::Parser;

use rustc_ast::token;
use rustc_ast::{
    self as ast, Attribute, GenericBounds, GenericParam, GenericParamKind, WhereClause,
};
use rustc_errors::PResult;
use rustc_span::symbol::{kw, sym};

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
    fn parse_ty_param(&mut self, preceding_attrs: Vec<Attribute>) -> PResult<'a, GenericParam> {
        let ident = self.parse_ident()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_generic_bounds(Some(self.prev_token.span))?
        } else {
            Vec::new()
        };

        let default = if self.eat(&token::Eq) { Some(self.parse_ty()?) } else { None };

        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs.into(),
            bounds,
            kind: GenericParamKind::Type { default },
            is_placeholder: false,
        })
    }

    fn parse_const_param(&mut self, preceding_attrs: Vec<Attribute>) -> PResult<'a, GenericParam> {
        let const_span = self.token.span;

        self.expect_keyword(kw::Const)?;
        let ident = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;

        // Parse optional const generics default value, taking care of feature gating the spans
        // with the unstable syntax mechanism.
        let default = if self.eat(&token::Eq) {
            // The gated span goes from the `=` to the end of the const argument that follows (and
            // which could be a block expression).
            let start = self.prev_token.span;
            let const_arg = self.parse_const_arg()?;
            let span = start.to(const_arg.value.span);
            self.sess.gated_spans.gate(sym::const_generics_defaults, span);
            Some(const_arg)
        } else {
            None
        };

        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs.into(),
            bounds: Vec::new(),
            kind: GenericParamKind::Const { ty, kw_span: const_span, default },
            is_placeholder: false,
        })
    }

    /// Parses a (possibly empty) list of lifetime and type parameters, possibly including
    /// a trailing comma and erroneous trailing attributes.
    pub(super) fn parse_generic_params(&mut self) -> PResult<'a, Vec<ast::GenericParam>> {
        let mut params = Vec::new();
        loop {
            let attrs = self.parse_outer_attributes()?;
            if self.check_lifetime() {
                let lifetime = self.expect_lifetime();
                // Parse lifetime parameter.
                let bounds =
                    if self.eat(&token::Colon) { self.parse_lt_param_bounds() } else { Vec::new() };
                params.push(ast::GenericParam {
                    ident: lifetime.ident,
                    id: lifetime.id,
                    attrs: attrs.into(),
                    bounds,
                    kind: ast::GenericParamKind::Lifetime,
                    is_placeholder: false,
                });
            } else if self.check_keyword(kw::Const) {
                // Parse const parameter.
                params.push(self.parse_const_param(attrs)?);
            } else if self.check_ident() {
                // Parse type parameter.
                params.push(self.parse_ty_param(attrs)?);
            } else if self.token.can_begin_type() {
                // Trying to write an associated type bound? (#26271)
                let snapshot = self.clone();
                match self.parse_ty_where_predicate() {
                    Ok(where_predicate) => {
                        self.struct_span_err(
                            where_predicate.span(),
                            "bounds on associated types do not belong here",
                        )
                        .span_label(where_predicate.span(), "belongs in `where` clause")
                        .emit();
                    }
                    Err(mut err) => {
                        err.cancel();
                        *self = snapshot;
                        break;
                    }
                }
            } else {
                // Check for trailing attributes and stop parsing.
                if !attrs.is_empty() {
                    if !params.is_empty() {
                        self.struct_span_err(
                            attrs[0].span,
                            "trailing attribute after generic parameter",
                        )
                        .span_label(attrs[0].span, "attributes must go before parameters")
                        .emit();
                    } else {
                        self.struct_span_err(attrs[0].span, "attribute without generic parameters")
                            .span_label(
                                attrs[0].span,
                                "attributes are only permitted when preceding parameters",
                            )
                            .emit();
                    }
                }
                break;
            }

            if !self.eat(&token::Comma) {
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
        let span_lo = self.token.span;
        let (params, span) = if self.eat_lt() {
            let params = self.parse_generic_params()?;
            self.expect_gt()?;
            (params, span_lo.to(self.prev_token.span))
        } else {
            (vec![], self.prev_token.span.shrink_to_hi())
        };
        Ok(ast::Generics {
            params,
            where_clause: WhereClause {
                has_where_token: false,
                predicates: Vec::new(),
                span: self.prev_token.span.shrink_to_hi(),
            },
            span,
        })
    }

    /// Parses an optional where-clause and places it in `generics`.
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// where T : Trait<U, V> + 'b, 'a : 'b
    /// ```
    pub(super) fn parse_where_clause(&mut self) -> PResult<'a, WhereClause> {
        let mut where_clause = WhereClause {
            has_where_token: false,
            predicates: Vec::new(),
            span: self.prev_token.span.shrink_to_hi(),
        };

        if !self.eat_keyword(kw::Where) {
            return Ok(where_clause);
        }
        where_clause.has_where_token = true;
        let lo = self.prev_token.span;

        // We are considering adding generics to the `where` keyword as an alternative higher-rank
        // parameter syntax (as in `where<'a>` or `where<T>`. To avoid that being a breaking
        // change we parse those generics now, but report an error.
        if self.choose_generics_over_qpath(0) {
            let generics = self.parse_generics()?;
            self.struct_span_err(
                generics.span,
                "generic parameters on `where` clauses are reserved for future use",
            )
            .span_label(generics.span, "currently unsupported")
            .emit();
        }

        loop {
            let lo = self.token.span;
            if self.check_lifetime() && self.look_ahead(1, |t| !t.is_like_plus()) {
                let lifetime = self.expect_lifetime();
                // Bounds starting with a colon are mandatory, but possibly empty.
                self.expect(&token::Colon)?;
                let bounds = self.parse_lt_param_bounds();
                where_clause.predicates.push(ast::WherePredicate::RegionPredicate(
                    ast::WhereRegionPredicate {
                        span: lo.to(self.prev_token.span),
                        lifetime,
                        bounds,
                    },
                ));
            } else if self.check_type() {
                where_clause.predicates.push(self.parse_ty_where_predicate()?);
            } else {
                break;
            }

            if !self.eat(&token::Comma) {
                break;
            }
        }

        where_clause.span = lo.to(self.prev_token.span);
        Ok(where_clause)
    }

    fn parse_ty_where_predicate(&mut self) -> PResult<'a, ast::WherePredicate> {
        let lo = self.token.span;
        // Parse optional `for<'a, 'b>`.
        // This `for` is parsed greedily and applies to the whole predicate,
        // the bounded type can have its own `for` applying only to it.
        // Examples:
        // * `for<'a> Trait1<'a>: Trait2<'a /* ok */>`
        // * `(for<'a> Trait1<'a>): Trait2<'a /* not ok */>`
        // * `for<'a> for<'b> Trait1<'a, 'b>: Trait2<'a /* ok */, 'b /* not ok */>`
        let lifetime_defs = self.parse_late_bound_lifetime_defs()?;

        // Parse type with mandatory colon and (possibly empty) bounds,
        // or with mandatory equality sign and the second type.
        let ty = self.parse_ty_for_where_clause()?;
        if self.eat(&token::Colon) {
            let bounds = self.parse_generic_bounds(Some(self.prev_token.span))?;
            Ok(ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                span: lo.to(self.prev_token.span),
                bound_generic_params: lifetime_defs,
                bounded_ty: ty,
                bounds,
            }))
        // FIXME: Decide what should be used here, `=` or `==`.
        // FIXME: We are just dropping the binders in lifetime_defs on the floor here.
        } else if self.eat(&token::Eq) || self.eat(&token::EqEq) {
            let rhs_ty = self.parse_ty()?;
            Ok(ast::WherePredicate::EqPredicate(ast::WhereEqPredicate {
                span: lo.to(self.prev_token.span),
                lhs_ty: ty,
                rhs_ty,
                id: ast::DUMMY_NODE_ID,
            }))
        } else {
            self.unexpected()
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
                    })
                || self.is_keyword_ahead(start + 1, &[kw::Const]))
    }
}
