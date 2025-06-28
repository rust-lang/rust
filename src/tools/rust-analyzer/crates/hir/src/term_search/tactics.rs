//! Tactics for term search
//!
//! All the tactics take following arguments
//! * `ctx` - Context for the term search
//! * `defs` - Set of items in scope at term search target location
//! * `lookup` - Lookup table for types
//! * `should_continue` - Function that indicates when to stop iterating
//!
//! And they return iterator that yields type trees that unify with the `goal` type.

use std::iter;

use hir_ty::TyBuilder;
use hir_ty::db::HirDatabase;
use hir_ty::mir::BorrowKind;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use span::Edition;

use crate::{
    Adt, AssocItem, GenericDef, GenericParam, HasAttrs, HasVisibility, Impl, ModuleDef, ScopeDef,
    Type, TypeParam,
};

use crate::term_search::Expr;

use super::{LookupTable, NewTypesKey, TermSearchCtx};

/// # Trivial tactic
///
/// Attempts to fulfill the goal by trying items in scope
/// Also works as a starting point to move all items in scope to lookup table.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
///
/// Returns iterator that yields elements that unify with `goal`.
///
/// _Note that there is no use of calling this tactic in every iteration as the output does not
/// depend on the current state of `lookup`_
pub(super) fn trivial<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    defs.iter().filter_map(|def| {
        let expr = match def {
            ScopeDef::ModuleDef(ModuleDef::Const(it)) => Some(Expr::Const(*it)),
            ScopeDef::ModuleDef(ModuleDef::Static(it)) => Some(Expr::Static(*it)),
            ScopeDef::GenericParam(GenericParam::ConstParam(it)) => Some(Expr::ConstParam(*it)),
            ScopeDef::Local(it) => {
                if ctx.config.enable_borrowcheck {
                    let borrowck = db.borrowck(it.parent).ok()?;

                    let invalid = borrowck.iter().any(|b| {
                        b.partially_moved.iter().any(|moved| {
                            Some(&moved.local) == b.mir_body.binding_locals.get(it.binding_id)
                        }) || b.borrow_regions.iter().any(|region| {
                            // Shared borrows are fine
                            Some(&region.local) == b.mir_body.binding_locals.get(it.binding_id)
                                && region.kind != BorrowKind::Shared
                        })
                    });

                    if invalid {
                        return None;
                    }
                }

                Some(Expr::Local(*it))
            }
            _ => None,
        }?;

        let ty = expr.ty(db);
        lookup.insert(ty.clone(), std::iter::once(expr.clone()));

        // Don't suggest local references as they are not valid for return
        if matches!(expr, Expr::Local(_))
            && ty.contains_reference(db)
            && ctx.config.enable_borrowcheck
        {
            return None;
        }

        ty.could_unify_with_deeply(db, &ctx.goal).then_some(expr)
    })
}

/// # Associated constant tactic
///
/// Attempts to fulfill the goal by trying constants defined as associated items.
/// Only considers them on types that are in scope.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
///
/// Returns iterator that yields elements that unify with `goal`.
///
/// _Note that there is no use of calling this tactic in every iteration as the output does not
/// depend on the current state of `lookup`_
pub(super) fn assoc_const<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();

    defs.iter()
        .filter_map(|def| match def {
            ScopeDef::ModuleDef(ModuleDef::Adt(it)) => Some(it),
            _ => None,
        })
        .flat_map(|it| Impl::all_for_type(db, it.ty(db)))
        .filter(|it| !it.is_unsafe(db))
        .flat_map(|it| it.items(db))
        .filter(move |it| it.is_visible_from(db, module))
        .filter_map(AssocItem::as_const)
        .filter_map(|it| {
            if it.attrs(db).is_unstable() {
                return None;
            }

            let expr = Expr::Const(it);
            let ty = it.ty(db);

            if ty.contains_unknown() {
                return None;
            }

            lookup.insert(ty.clone(), std::iter::once(expr.clone()));

            ty.could_unify_with_deeply(db, &ctx.goal).then_some(expr)
        })
}

/// # Data constructor tactic
///
/// Attempts different data constructors for enums and structs in scope
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `should_continue` - Function that indicates when to stop iterating
pub(super) fn data_constructor<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
    should_continue: &'a dyn std::ops::Fn() -> bool,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .types_wishlist()
        .clone()
        .into_iter()
        .chain(iter::once(ctx.goal.clone()))
        .filter_map(|ty| ty.as_adt().map(|adt| (adt, ty)))
        .filter(|_| should_continue())
        .filter_map(move |(adt, ty)| match adt {
            Adt::Struct(strukt) => {
                // Ignore unstable or not visible
                if strukt.is_unstable(db) || !strukt.is_visible_from(db, module) {
                    return None;
                }

                let generics = GenericDef::from(strukt);

                // We currently do not check lifetime bounds so ignore all types that have something to do
                // with them
                if !generics.lifetime_params(db).is_empty() {
                    return None;
                }

                if ty.contains_unknown() {
                    return None;
                }

                // Ignore types that have something to do with lifetimes
                if ctx.config.enable_borrowcheck && ty.contains_reference(db) {
                    return None;
                }
                let fields = strukt.fields(db);
                // Check if all fields are visible, otherwise we cannot fill them
                if fields.iter().any(|it| !it.is_visible_from(db, module)) {
                    return None;
                }

                let generics: Vec<_> = ty.type_arguments().collect();

                // Early exit if some param cannot be filled from lookup
                let param_exprs: Vec<Vec<Expr<'_>>> = fields
                    .into_iter()
                    .map(|field| lookup.find(db, &field.ty_with_args(db, generics.iter().cloned())))
                    .collect::<Option<_>>()?;

                // Note that we need special case for 0 param constructors because of multi cartesian
                // product
                let exprs: Vec<Expr<'_>> = if param_exprs.is_empty() {
                    vec![Expr::Struct { strukt, generics, params: Vec::new() }]
                } else {
                    param_exprs
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|params| Expr::Struct { strukt, generics: generics.clone(), params })
                        .collect()
                };

                lookup.insert(ty.clone(), exprs.iter().cloned());
                Some((ty, exprs))
            }
            Adt::Enum(enum_) => {
                // Ignore unstable or not visible
                if enum_.is_unstable(db) || !enum_.is_visible_from(db, module) {
                    return None;
                }

                let generics = GenericDef::from(enum_);
                // We currently do not check lifetime bounds so ignore all types that have something to do
                // with them
                if !generics.lifetime_params(db).is_empty() {
                    return None;
                }

                if ty.contains_unknown() {
                    return None;
                }

                // Ignore types that have something to do with lifetimes
                if ctx.config.enable_borrowcheck && ty.contains_reference(db) {
                    return None;
                }

                let generics: Vec<_> = ty.type_arguments().collect();
                let exprs = enum_
                    .variants(db)
                    .into_iter()
                    .filter_map(|variant| {
                        // Early exit if some param cannot be filled from lookup
                        let param_exprs: Vec<Vec<Expr<'_>>> = variant
                            .fields(db)
                            .into_iter()
                            .map(|field| {
                                lookup.find(db, &field.ty_with_args(db, generics.iter().cloned()))
                            })
                            .collect::<Option<_>>()?;

                        // Note that we need special case for 0 param constructors because of multi cartesian
                        // product
                        let variant_exprs: Vec<Expr<'_>> = if param_exprs.is_empty() {
                            vec![Expr::Variant {
                                variant,
                                generics: generics.clone(),
                                params: Vec::new(),
                            }]
                        } else {
                            param_exprs
                                .into_iter()
                                .multi_cartesian_product()
                                .map(|params| Expr::Variant {
                                    variant,
                                    generics: generics.clone(),
                                    params,
                                })
                                .collect()
                        };
                        lookup.insert(ty.clone(), variant_exprs.iter().cloned());
                        Some(variant_exprs)
                    })
                    .flatten()
                    .collect();

                Some((ty, exprs))
            }
            Adt::Union(_) => None,
        })
        .filter_map(|(ty, exprs)| ty.could_unify_with_deeply(db, &ctx.goal).then_some(exprs))
        .flatten()
}

/// # Free function tactic
///
/// Attempts to call different functions in scope with parameters from lookup table.
/// Functions that include generics are not used for performance reasons.
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `should_continue` - Function that indicates when to stop iterating
pub(super) fn free_function<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
    should_continue: &'a dyn std::ops::Fn() -> bool,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    defs.iter()
        .filter_map(move |def| match def {
            ScopeDef::ModuleDef(ModuleDef::Function(it)) => {
                let generics = GenericDef::from(*it);

                // Ignore const params for now
                let type_params = generics
                    .type_or_const_params(db)
                    .into_iter()
                    .map(|it| it.as_type_param(db))
                    .collect::<Option<Vec<TypeParam>>>()?;

                // Ignore lifetimes as we do not check them
                if !generics.lifetime_params(db).is_empty() {
                    return None;
                }

                // Only account for stable type parameters for now, unstable params can be default
                // tho, for example in `Box<T, #[unstable] A: Allocator>`
                if type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none()) {
                    return None;
                }

                let non_default_type_params_len =
                    type_params.iter().filter(|it| it.default(db).is_none()).count();

                // Ignore bigger number of generics for now as they kill the performance
                if non_default_type_params_len > 0 {
                    return None;
                }

                let generic_params = lookup
                    .iter_types()
                    .collect::<Vec<_>>() // Force take ownership
                    .into_iter()
                    .permutations(non_default_type_params_len);

                let exprs: Vec<_> = generic_params
                    .filter(|_| should_continue())
                    .filter_map(|generics| {
                        // Insert default type params
                        let mut g = generics.into_iter();
                        let generics: Vec<_> = type_params
                            .iter()
                            .map(|it| match it.default(db) {
                                Some(ty) => Some(ty),
                                None => {
                                    let generic = g.next().expect("Missing type param");
                                    // Filter out generics that do not unify due to trait bounds
                                    it.ty(db).could_unify_with(db, &generic).then_some(generic)
                                }
                            })
                            .collect::<Option<_>>()?;

                        let ret_ty = it.ret_type_with_args(db, generics.iter().cloned());
                        // Filter out private and unsafe functions
                        if !it.is_visible_from(db, module)
                            || it.is_unsafe_to_call(db, None, Edition::CURRENT_FIXME)
                            || it.is_unstable(db)
                            || ctx.config.enable_borrowcheck && ret_ty.contains_reference(db)
                            || ret_ty.is_raw_ptr()
                        {
                            return None;
                        }

                        // Early exit if some param cannot be filled from lookup
                        let param_exprs: Vec<Vec<Expr<'_>>> = it
                            .params_without_self_with_args(db, generics.iter().cloned())
                            .into_iter()
                            .map(|field| {
                                let ty = field.ty();
                                match ty.is_mutable_reference() {
                                    true => None,
                                    false => lookup.find_autoref(db, ty),
                                }
                            })
                            .collect::<Option<_>>()?;

                        // Note that we need special case for 0 param constructors because of multi cartesian
                        // product
                        let fn_exprs: Vec<Expr<'_>> = if param_exprs.is_empty() {
                            vec![Expr::Function { func: *it, generics, params: Vec::new() }]
                        } else {
                            param_exprs
                                .into_iter()
                                .multi_cartesian_product()
                                .map(|params| Expr::Function {
                                    func: *it,
                                    generics: generics.clone(),

                                    params,
                                })
                                .collect()
                        };

                        lookup.insert(ret_ty.clone(), fn_exprs.iter().cloned());
                        Some((ret_ty, fn_exprs))
                    })
                    .collect();
                Some(exprs)
            }
            _ => None,
        })
        .flatten()
        .filter_map(|(ty, exprs)| ty.could_unify_with_deeply(db, &ctx.goal).then_some(exprs))
        .flatten()
}

/// # Impl method tactic
///
/// Attempts to call methods on types from lookup table.
/// This includes both functions from direct impl blocks as well as functions from traits.
/// Methods defined in impl blocks that are generic and methods that are themselves have
/// generics are ignored for performance reasons.
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `should_continue` - Function that indicates when to stop iterating
pub(super) fn impl_method<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
    should_continue: &'a dyn std::ops::Fn() -> bool,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .new_types(NewTypesKey::ImplMethod)
        .into_iter()
        .filter(|ty| !ty.type_arguments().any(|it| it.contains_unknown()))
        .filter(|_| should_continue())
        .flat_map(|ty| {
            Impl::all_for_type(db, ty.clone()).into_iter().map(move |imp| (ty.clone(), imp))
        })
        .flat_map(|(ty, imp)| imp.items(db).into_iter().map(move |item| (imp, ty.clone(), item)))
        .filter_map(|(imp, ty, it)| match it {
            AssocItem::Function(f) => Some((imp, ty, f)),
            _ => None,
        })
        .filter(|_| should_continue())
        .filter_map(move |(imp, ty, it)| {
            let fn_generics = GenericDef::from(it);
            let imp_generics = GenericDef::from(imp);

            // Ignore all functions that have something to do with lifetimes as we don't check them
            if !fn_generics.lifetime_params(db).is_empty()
                || !imp_generics.lifetime_params(db).is_empty()
            {
                return None;
            }

            // Ignore functions without self param
            if !it.has_self_param(db) {
                return None;
            }

            // Filter out private and unsafe functions
            if !it.is_visible_from(db, module)
                || it.is_unsafe_to_call(db, None, Edition::CURRENT_FIXME)
                || it.is_unstable(db)
            {
                return None;
            }

            // Ignore functions with generics for now as they kill the performance
            // Also checking bounds for generics is problematic
            if !fn_generics.type_or_const_params(db).is_empty() {
                return None;
            }

            let ret_ty = it.ret_type_with_args(db, ty.type_arguments());
            // Filter out functions that return references
            if ctx.config.enable_borrowcheck && ret_ty.contains_reference(db) || ret_ty.is_raw_ptr()
            {
                return None;
            }

            // Ignore functions that do not change the type
            if ty.could_unify_with_deeply(db, &ret_ty) {
                return None;
            }

            let self_ty =
                it.self_param(db).expect("No self param").ty_with_args(db, ty.type_arguments());

            // Ignore functions that have different self type
            if !self_ty.autoderef(db).any(|s_ty| ty == s_ty) {
                return None;
            }

            let target_type_exprs = lookup.find(db, &ty).expect("Type not in lookup");

            // Early exit if some param cannot be filled from lookup
            let param_exprs: Vec<Vec<Expr<'_>>> = it
                .params_without_self_with_args(db, ty.type_arguments())
                .into_iter()
                .map(|field| lookup.find_autoref(db, field.ty()))
                .collect::<Option<_>>()?;

            let generics: Vec<_> = ty.type_arguments().collect();
            let fn_exprs: Vec<Expr<'_>> = std::iter::once(target_type_exprs)
                .chain(param_exprs)
                .multi_cartesian_product()
                .map(|params| {
                    let mut params = params.into_iter();
                    let target = Box::new(params.next().unwrap());
                    Expr::Method {
                        func: it,
                        generics: generics.clone(),
                        target,
                        params: params.collect(),
                    }
                })
                .collect();

            Some((ret_ty, fn_exprs))
        })
        .filter_map(|(ty, exprs)| ty.could_unify_with_deeply(db, &ctx.goal).then_some(exprs))
        .flatten()
}

/// # Struct projection tactic
///
/// Attempts different struct fields (`foo.bar.baz`)
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `should_continue` - Function that indicates when to stop iterating
pub(super) fn struct_projection<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
    should_continue: &'a dyn std::ops::Fn() -> bool,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .new_types(NewTypesKey::StructProjection)
        .into_iter()
        .map(|ty| (ty.clone(), lookup.find(db, &ty).expect("Expr not in lookup")))
        .filter(|_| should_continue())
        .flat_map(move |(ty, targets)| {
            ty.fields(db).into_iter().filter_map(move |(field, filed_ty)| {
                if !field.is_visible_from(db, module) {
                    return None;
                }
                let exprs = targets
                    .clone()
                    .into_iter()
                    .map(move |target| Expr::Field { field, expr: Box::new(target) });
                Some((filed_ty, exprs))
            })
        })
        .filter_map(|(ty, exprs)| ty.could_unify_with_deeply(db, &ctx.goal).then_some(exprs))
        .flatten()
}

/// # Famous types tactic
///
/// Attempts different values of well known types such as `true` or `false`.
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// _Note that there is no point of calling it iteratively as the output is always the same_
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
pub(super) fn famous_types<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    [
        Expr::FamousType { ty: Type::new(db, module.id, TyBuilder::bool()), value: "true" },
        Expr::FamousType { ty: Type::new(db, module.id, TyBuilder::bool()), value: "false" },
        Expr::FamousType { ty: Type::new(db, module.id, TyBuilder::unit()), value: "()" },
    ]
    .into_iter()
    .inspect(|exprs| {
        lookup.insert(exprs.ty(db), std::iter::once(exprs.clone()));
    })
    .filter(|expr| expr.ty(db).could_unify_with_deeply(db, &ctx.goal))
}

/// # Impl static method (without self type) tactic
///
/// Attempts different functions from impl blocks that take no self parameter.
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `should_continue` - Function that indicates when to stop iterating
pub(super) fn impl_static_method<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
    should_continue: &'a dyn std::ops::Fn() -> bool,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .types_wishlist()
        .clone()
        .into_iter()
        .chain(iter::once(ctx.goal.clone()))
        .filter(|ty| !ty.type_arguments().any(|it| it.contains_unknown()))
        .filter(|_| should_continue())
        .flat_map(|ty| {
            Impl::all_for_type(db, ty.clone()).into_iter().map(move |imp| (ty.clone(), imp))
        })
        .filter(|(_, imp)| !imp.is_unsafe(db))
        .flat_map(|(ty, imp)| imp.items(db).into_iter().map(move |item| (imp, ty.clone(), item)))
        .filter_map(|(imp, ty, it)| match it {
            AssocItem::Function(f) => Some((imp, ty, f)),
            _ => None,
        })
        .filter(|_| should_continue())
        .filter_map(move |(imp, ty, it)| {
            let fn_generics = GenericDef::from(it);
            let imp_generics = GenericDef::from(imp);

            // Ignore all functions that have something to do with lifetimes as we don't check them
            if !fn_generics.lifetime_params(db).is_empty()
                || !imp_generics.lifetime_params(db).is_empty()
            {
                return None;
            }

            // Ignore functions with self param
            if it.has_self_param(db) {
                return None;
            }

            // Filter out private and unsafe functions
            if !it.is_visible_from(db, module)
                || it.is_unsafe_to_call(db, None, Edition::CURRENT_FIXME)
                || it.is_unstable(db)
            {
                return None;
            }

            // Ignore functions with generics for now as they kill the performance
            // Also checking bounds for generics is problematic
            if !fn_generics.type_or_const_params(db).is_empty() {
                return None;
            }

            let ret_ty = it.ret_type_with_args(db, ty.type_arguments());
            // Filter out functions that return references
            if ctx.config.enable_borrowcheck && ret_ty.contains_reference(db) || ret_ty.is_raw_ptr()
            {
                return None;
            }

            // Early exit if some param cannot be filled from lookup
            let param_exprs: Vec<Vec<Expr<'_>>> = it
                .params_without_self_with_args(db, ty.type_arguments())
                .into_iter()
                .map(|field| lookup.find_autoref(db, field.ty()))
                .collect::<Option<_>>()?;

            // Note that we need special case for 0 param constructors because of multi cartesian
            // product
            let generics = ty.type_arguments().collect();
            let fn_exprs: Vec<Expr<'_>> = if param_exprs.is_empty() {
                vec![Expr::Function { func: it, generics, params: Vec::new() }]
            } else {
                param_exprs
                    .into_iter()
                    .multi_cartesian_product()
                    .map(|params| Expr::Function { func: it, generics: generics.clone(), params })
                    .collect()
            };

            lookup.insert(ret_ty.clone(), fn_exprs.iter().cloned());

            Some((ret_ty, fn_exprs))
        })
        .filter_map(|(ty, exprs)| ty.could_unify_with_deeply(db, &ctx.goal).then_some(exprs))
        .flatten()
}

/// # Make tuple tactic
///
/// Attempts to create tuple types if any are listed in types wishlist
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `should_continue` - Function that indicates when to stop iterating
pub(super) fn make_tuple<'a, 'lt, 'db, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'db, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'lt mut LookupTable<'db>,
    should_continue: &'a dyn std::ops::Fn() -> bool,
) -> impl Iterator<Item = Expr<'db>> + use<'a, 'db, 'lt, DB> {
    let db = ctx.sema.db;
    let module = ctx.scope.module();

    lookup
        .types_wishlist()
        .clone()
        .into_iter()
        .filter(|_| should_continue())
        .filter(|ty| ty.is_tuple())
        .filter_map(move |ty| {
            // Double check to not contain unknown
            if ty.contains_unknown() {
                return None;
            }

            // Ignore types that have something to do with lifetimes
            if ctx.config.enable_borrowcheck && ty.contains_reference(db) {
                return None;
            }

            // Early exit if some param cannot be filled from lookup
            let param_exprs: Vec<Vec<Expr<'db>>> =
                ty.type_arguments().map(|field| lookup.find(db, &field)).collect::<Option<_>>()?;

            let exprs: Vec<Expr<'db>> = param_exprs
                .into_iter()
                .multi_cartesian_product()
                .filter(|_| should_continue())
                .map(|params| {
                    let tys: Vec<Type<'_>> = params.iter().map(|it| it.ty(db)).collect();
                    let tuple_ty = Type::new_tuple(module.krate().into(), &tys);

                    let expr = Expr::Tuple { ty: tuple_ty.clone(), params };
                    lookup.insert(tuple_ty, iter::once(expr.clone()));
                    expr
                })
                .collect();

            Some(exprs)
        })
        .flatten()
        .filter_map(|expr| expr.ty(db).could_unify_with_deeply(db, &ctx.goal).then_some(expr))
}
