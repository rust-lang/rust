use ast::HasAttrs;
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast::visit::BoundKind;
use rustc_ast::{
    self as ast, GenericArg, GenericBound, GenericParamKind, Generics, ItemKind, MetaItem,
    TraitBoundModifiers, VariantData, WherePredicate,
};
use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_errors::E0802;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_macros::Diagnostic;
use rustc_span::{Ident, Span, Symbol, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::errors;

macro_rules! path {
    ($span:expr, $($part:ident)::*) => { vec![$(Ident::new(sym::$part, $span),)*] }
}

pub(crate) fn expand_deriving_coerce_pointee(
    cx: &ExtCtxt<'_>,
    span: Span,
    _mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    _is_const: bool,
) {
    item.visit_with(&mut DetectNonGenericPointeeAttr { cx });

    let (name_ident, generics) = if let Annotatable::Item(aitem) = item
        && let ItemKind::Struct(ident, g, struct_data) = &aitem.kind
    {
        if !matches!(
            struct_data,
            VariantData::Struct { fields, recovered: _ } | VariantData::Tuple(fields, _)
                if !fields.is_empty())
        {
            cx.dcx().emit_err(RequireOneField { span });
            return;
        }
        (*ident, g)
    } else {
        cx.dcx().emit_err(RequireTransparent { span });
        return;
    };

    // Convert generic parameters (from the struct) into generic args.
    let self_params: Vec<_> = generics
        .params
        .iter()
        .map(|p| match p.kind {
            GenericParamKind::Lifetime => GenericArg::Lifetime(cx.lifetime(p.span(), p.ident)),
            GenericParamKind::Type { .. } => GenericArg::Type(cx.ty_ident(p.span(), p.ident)),
            GenericParamKind::Const { .. } => GenericArg::Const(cx.const_ident(p.span(), p.ident)),
        })
        .collect();
    let type_params: Vec<_> = generics
        .params
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| {
            if let GenericParamKind::Type { .. } = p.kind {
                Some((idx, p.span(), p.attrs().iter().any(|attr| attr.has_name(sym::pointee))))
            } else {
                None
            }
        })
        .collect();

    let pointee_param_idx = if type_params.is_empty() {
        // `#[derive(CoercePointee)]` requires at least one generic type on the target `struct`
        cx.dcx().emit_err(RequireOneGeneric { span });
        return;
    } else if type_params.len() == 1 {
        // Regardless of the only type param being designed as `#[pointee]` or not, we can just use it as such
        type_params[0].0
    } else {
        let mut pointees = type_params
            .iter()
            .filter_map(|&(idx, span, is_pointee)| is_pointee.then_some((idx, span)));
        match (pointees.next(), pointees.next()) {
            (Some((idx, _span)), None) => idx,
            (None, _) => {
                cx.dcx().emit_err(RequireOnePointee { span });
                return;
            }
            (Some((_, one)), Some((_, another))) => {
                cx.dcx().emit_err(TooManyPointees { one, another });
                return;
            }
        }
    };

    // Create the type of `self`.
    let path = cx.path_all(span, false, vec![name_ident], self_params.clone());
    let self_type = cx.ty_path(path);

    // Declare helper function that adds implementation blocks.
    // FIXME(dingxiangfei2009): Investigate the set of attributes on target struct to be propagated to impls
    let attrs = thin_vec![cx.attr_word(sym::automatically_derived, span),];
    // # Validity assertion which will be checked later in `rustc_hir_analysis::coherence::builtins`.
    {
        let trait_path =
            cx.path_all(span, true, path!(span, core::marker::CoercePointeeValidated), vec![]);
        let trait_ref = cx.trait_ref(trait_path);
        push(Annotatable::Item(
            cx.item(
                span,
                attrs.clone(),
                ast::ItemKind::Impl(ast::Impl {
                    generics: Generics {
                        params: generics
                            .params
                            .iter()
                            .map(|p| match &p.kind {
                                GenericParamKind::Lifetime => {
                                    cx.lifetime_param(p.span(), p.ident, p.bounds.clone())
                                }
                                GenericParamKind::Type { default: _ } => {
                                    cx.typaram(p.span(), p.ident, p.bounds.clone(), None)
                                }
                                GenericParamKind::Const { ty, span: _, default: _ } => cx
                                    .const_param(
                                        p.span(),
                                        p.ident,
                                        p.bounds.clone(),
                                        ty.clone(),
                                        None,
                                    ),
                            })
                            .collect(),
                        where_clause: generics.where_clause.clone(),
                        span: generics.span,
                    },
                    of_trait: Some(Box::new(ast::TraitImplHeader {
                        safety: ast::Safety::Default,
                        polarity: ast::ImplPolarity::Positive,
                        defaultness: ast::Defaultness::Final,
                        constness: ast::Const::No,
                        trait_ref,
                    })),
                    self_ty: self_type.clone(),
                    items: ThinVec::new(),
                }),
            ),
        ));
    }
    let mut add_impl_block = |generics, trait_symbol, trait_args| {
        let mut parts = path!(span, core::ops);
        parts.push(Ident::new(trait_symbol, span));
        let trait_path = cx.path_all(span, true, parts, trait_args);
        let trait_ref = cx.trait_ref(trait_path);
        let item = cx.item(
            span,
            attrs.clone(),
            ast::ItemKind::Impl(ast::Impl {
                generics,
                of_trait: Some(Box::new(ast::TraitImplHeader {
                    safety: ast::Safety::Default,
                    polarity: ast::ImplPolarity::Positive,
                    defaultness: ast::Defaultness::Final,
                    constness: ast::Const::No,
                    trait_ref,
                })),
                self_ty: self_type.clone(),
                items: ThinVec::new(),
            }),
        );
        push(Annotatable::Item(item));
    };

    // Create unsized `self`, that is, one where the `#[pointee]` type arg is replaced with `__S`. For
    // example, instead of `MyType<'a, T>`, it will be `MyType<'a, __S>`.
    let s_ty = cx.ty_ident(span, Ident::new(sym::__S, span));
    let mut alt_self_params = self_params;
    alt_self_params[pointee_param_idx] = GenericArg::Type(s_ty.clone());
    let alt_self_type = cx.ty_path(cx.path_all(span, false, vec![name_ident], alt_self_params));

    // # Add `Unsize<__S>` bound to `#[pointee]` at the generic parameter location
    //
    // Find the `#[pointee]` parameter and add an `Unsize<__S>` bound to it.
    let mut impl_generics = generics.clone();
    let pointee_ty_ident = generics.params[pointee_param_idx].ident;
    let mut self_bounds;
    {
        let pointee = &mut impl_generics.params[pointee_param_idx];
        self_bounds = pointee.bounds.clone();
        if !contains_maybe_sized_bound(&self_bounds)
            && !contains_maybe_sized_bound_on_pointee(
                &generics.where_clause.predicates,
                pointee_ty_ident.name,
            )
        {
            cx.dcx().emit_err(RequiresMaybeSized {
                span: pointee_ty_ident.span,
                name: pointee_ty_ident,
            });
            return;
        }
        let arg = GenericArg::Type(s_ty.clone());
        let unsize = cx.path_all(span, true, path!(span, core::marker::Unsize), vec![arg]);
        pointee.bounds.push(cx.trait_bound(unsize, false));
        // Drop `#[pointee]` attribute since it should not be recognized outside `derive(CoercePointee)`
        pointee.attrs.retain(|attr| !attr.has_name(sym::pointee));
    }

    // # Rewrite generic parameter bounds
    // For each bound `U: ..` in `struct<U: ..>`, make a new bound with `__S` in place of `#[pointee]`
    // Example:
    // ```
    // struct<
    //     U: Trait<T>,
    //     #[pointee] T: Trait<T> + ?Sized,
    //     V: Trait<T>> ...
    // ```
    // ... generates this `impl` generic parameters
    // ```
    // impl<
    //     U: Trait<T> + Trait<__S>,
    //     T: Trait<T> + ?Sized + Unsize<__S>, // (**)
    //     __S: Trait<__S> + ?Sized, // (*)
    //     V: Trait<T> + Trait<__S>> ...
    // ```
    // The new bound marked with (*) has to be done separately.
    // See next section
    for (idx, (params, orig_params)) in
        impl_generics.params.iter_mut().zip(&generics.params).enumerate()
    {
        // Default type parameters are rejected for `impl` block.
        // We should drop them now.
        match &mut params.kind {
            ast::GenericParamKind::Const { default, .. } => *default = None,
            ast::GenericParamKind::Type { default } => *default = None,
            ast::GenericParamKind::Lifetime => {}
        }
        // We CANNOT rewrite `#[pointee]` type parameter bounds.
        // This has been set in stone. (**)
        // So we skip over it.
        // Otherwise, we push extra bounds involving `__S`.
        if idx != pointee_param_idx {
            for bound in &orig_params.bounds {
                let mut bound = bound.clone();
                let mut substitution = TypeSubstitution {
                    from_name: pointee_ty_ident.name,
                    to_ty: &s_ty,
                    rewritten: false,
                };
                substitution.visit_param_bound(&mut bound, BoundKind::Bound);
                if substitution.rewritten {
                    // We found use of `#[pointee]` somewhere,
                    // so we make a new bound using `__S` in place of `#[pointee]`
                    params.bounds.push(bound);
                }
            }
        }
    }

    // # Insert `__S` type parameter
    //
    // We now insert `__S` with the missing bounds marked with (*) above.
    // We should also write the bounds from `#[pointee]` to `__S` as required by `Unsize<__S>`.
    {
        let mut substitution =
            TypeSubstitution { from_name: pointee_ty_ident.name, to_ty: &s_ty, rewritten: false };
        for bound in &mut self_bounds {
            substitution.visit_param_bound(bound, BoundKind::Bound);
        }
    }

    // # Rewrite `where` clauses
    //
    // Move on to `where` clauses.
    // Example:
    // ```
    // struct MyPointer<#[pointee] T, ..>
    // where
    //   U: Trait<V> + Trait<T>,
    //   Companion<T>: Trait<T>,
    //   T: Trait<T> + ?Sized,
    // { .. }
    // ```
    // ... will have a impl prelude like so
    // ```
    // impl<..> ..
    // where
    //   U: Trait<V> + Trait<T>,
    //   U: Trait<__S>,
    //   Companion<T>: Trait<T>,
    //   Companion<__S>: Trait<__S>,
    //   T: Trait<T> + ?Sized,
    //   __S: Trait<__S> + ?Sized,
    // ```
    //
    // We should also write a few new `where` bounds from `#[pointee] T` to `__S`
    // as well as any bound that indirectly involves the `#[pointee] T` type.
    for predicate in &generics.where_clause.predicates {
        if let ast::WherePredicateKind::BoundPredicate(bound) = &predicate.kind {
            let mut substitution = TypeSubstitution {
                from_name: pointee_ty_ident.name,
                to_ty: &s_ty,
                rewritten: false,
            };
            let mut kind = ast::WherePredicateKind::BoundPredicate(bound.clone());
            substitution.visit_where_predicate_kind(&mut kind);
            if substitution.rewritten {
                let predicate = ast::WherePredicate {
                    attrs: predicate.attrs.clone(),
                    kind,
                    span: predicate.span,
                    id: ast::DUMMY_NODE_ID,
                    is_placeholder: false,
                };
                impl_generics.where_clause.predicates.push(predicate);
            }
        }
    }

    let extra_param = cx.typaram(span, Ident::new(sym::__S, span), self_bounds, None);
    impl_generics.params.insert(pointee_param_idx + 1, extra_param);

    // Add the impl blocks for `DispatchFromDyn` and `CoerceUnsized`.
    let gen_args = vec![GenericArg::Type(alt_self_type)];
    add_impl_block(impl_generics.clone(), sym::DispatchFromDyn, gen_args.clone());
    add_impl_block(impl_generics.clone(), sym::CoerceUnsized, gen_args);
}

fn contains_maybe_sized_bound_on_pointee(predicates: &[WherePredicate], pointee: Symbol) -> bool {
    for bound in predicates {
        if let ast::WherePredicateKind::BoundPredicate(bound) = &bound.kind
            && bound.bounded_ty.kind.is_simple_path().is_some_and(|name| name == pointee)
        {
            for bound in &bound.bounds {
                if is_maybe_sized_bound(bound) {
                    return true;
                }
            }
        }
    }
    false
}

fn is_maybe_sized_bound(bound: &GenericBound) -> bool {
    if let GenericBound::Trait(trait_ref) = bound
        && let TraitBoundModifiers { polarity: ast::BoundPolarity::Maybe(_), .. } =
            trait_ref.modifiers
        && is_sized_marker(&trait_ref.trait_ref.path)
    {
        true
    } else {
        false
    }
}

fn contains_maybe_sized_bound(bounds: &[GenericBound]) -> bool {
    bounds.iter().any(is_maybe_sized_bound)
}

fn is_sized_marker(path: &ast::Path) -> bool {
    const CORE_UNSIZE: [Symbol; 3] = [sym::core, sym::marker, sym::Sized];
    const STD_UNSIZE: [Symbol; 3] = [sym::std, sym::marker, sym::Sized];
    let segments = || path.segments.iter().map(|segment| segment.ident.name);
    if path.is_global() {
        segments().skip(1).eq(CORE_UNSIZE) || segments().skip(1).eq(STD_UNSIZE)
    } else {
        segments().eq(CORE_UNSIZE) || segments().eq(STD_UNSIZE) || *path == sym::Sized
    }
}

struct TypeSubstitution<'a> {
    from_name: Symbol,
    to_ty: &'a ast::Ty,
    rewritten: bool,
}

impl<'a> ast::mut_visit::MutVisitor for TypeSubstitution<'a> {
    fn visit_ty(&mut self, ty: &mut ast::Ty) {
        if let Some(name) = ty.kind.is_simple_path()
            && name == self.from_name
        {
            *ty = self.to_ty.clone();
            self.rewritten = true;
        } else {
            ast::mut_visit::walk_ty(self, ty);
        }
    }

    fn visit_where_predicate_kind(&mut self, kind: &mut ast::WherePredicateKind) {
        match kind {
            rustc_ast::WherePredicateKind::BoundPredicate(bound) => {
                bound
                    .bound_generic_params
                    .flat_map_in_place(|param| self.flat_map_generic_param(param));
                self.visit_ty(&mut bound.bounded_ty);
                for bound in &mut bound.bounds {
                    self.visit_param_bound(bound, BoundKind::Bound)
                }
            }
            rustc_ast::WherePredicateKind::RegionPredicate(_)
            | rustc_ast::WherePredicateKind::EqPredicate(_) => {}
        }
    }
}

struct DetectNonGenericPointeeAttr<'a, 'b> {
    cx: &'a ExtCtxt<'b>,
}

impl<'a, 'b> rustc_ast::visit::Visitor<'a> for DetectNonGenericPointeeAttr<'a, 'b> {
    fn visit_attribute(&mut self, attr: &'a rustc_ast::Attribute) -> Self::Result {
        if attr.has_name(sym::pointee) {
            self.cx.dcx().emit_err(errors::NonGenericPointee { span: attr.span });
        }
    }

    fn visit_generic_param(&mut self, param: &'a rustc_ast::GenericParam) -> Self::Result {
        let mut error_on_pointee = AlwaysErrorOnGenericParam { cx: self.cx };

        match &param.kind {
            GenericParamKind::Type { default } => {
                // The `default` may end up containing a block expression.
                // The problem is block expressions  may define structs with generics.
                // A user may attach a #[pointee] attribute to one of these generics
                // We want to catch that. The simple solution is to just
                // always raise a `NonGenericPointee` error when this happens.
                //
                // This solution does reject valid rust programs but,
                // such a code would have to, in order:
                // - Define a smart pointer struct.
                // - Somewhere in this struct definition use a type with a const generic argument.
                // - Calculate this const generic in a expression block.
                // - Define a new smart pointer type in this block.
                // - Have this smart pointer type have more than 1 generic type.
                // In this case, the inner smart pointer derive would be complaining that it
                // needs a pointer attribute. Meanwhile, the outer macro would be complaining
                // that we attached a #[pointee] to a generic type argument while helpfully
                // informing the user that #[pointee] can only be attached to generic pointer arguments
                rustc_ast::visit::visit_opt!(error_on_pointee, visit_ty, default);
            }

            GenericParamKind::Const { .. } | GenericParamKind::Lifetime => {
                rustc_ast::visit::walk_generic_param(&mut error_on_pointee, param);
            }
        }
    }

    fn visit_ty(&mut self, t: &'a rustc_ast::Ty) -> Self::Result {
        let mut error_on_pointee = AlwaysErrorOnGenericParam { cx: self.cx };
        error_on_pointee.visit_ty(t)
    }
}

struct AlwaysErrorOnGenericParam<'a, 'b> {
    cx: &'a ExtCtxt<'b>,
}

impl<'a, 'b> rustc_ast::visit::Visitor<'a> for AlwaysErrorOnGenericParam<'a, 'b> {
    fn visit_attribute(&mut self, attr: &'a rustc_ast::Attribute) -> Self::Result {
        if attr.has_name(sym::pointee) {
            self.cx.dcx().emit_err(errors::NonGenericPointee { span: attr.span });
        }
    }
}

#[derive(Diagnostic)]
#[diag(builtin_macros_coerce_pointee_requires_transparent, code = E0802)]
struct RequireTransparent {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_coerce_pointee_requires_one_field, code = E0802)]
struct RequireOneField {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_coerce_pointee_requires_one_generic, code = E0802)]
struct RequireOneGeneric {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_coerce_pointee_requires_one_pointee, code = E0802)]
struct RequireOnePointee {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_coerce_pointee_too_many_pointees, code = E0802)]
struct TooManyPointees {
    #[primary_span]
    one: Span,
    #[label]
    another: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_coerce_pointee_requires_maybe_sized, code = E0802)]
struct RequiresMaybeSized {
    #[primary_span]
    span: Span,
    name: Ident,
}
