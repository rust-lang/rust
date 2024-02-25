//! Tactics for term search
//!
//! All the tactics take following arguments
//! * `ctx` - Context for the term search
//! * `defs` - Set of items in scope at term search target location
//! * `lookup` - Lookup table for types
//! And they return iterator that yields type trees that unify with the `goal` type.

use std::iter;

use hir_ty::db::HirDatabase;
use hir_ty::mir::BorrowKind;
use hir_ty::TyBuilder;
use itertools::Itertools;
use rustc_hash::FxHashSet;

use crate::{
    Adt, AssocItem, Enum, GenericDef, GenericParam, HasVisibility, Impl, ModuleDef, ScopeDef, Type,
    TypeParam, Variant,
};

use crate::term_search::{Expr, TermSearchConfig};

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
pub(super) fn trivial<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
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

        lookup.mark_exhausted(*def);

        let ty = expr.ty(db);
        lookup.insert(ty.clone(), std::iter::once(expr.clone()));

        // Don't suggest local references as they are not valid for return
        if matches!(expr, Expr::Local(_)) && ty.contains_reference(db) {
            return None;
        }

        ty.could_unify_with_deeply(db, &ctx.goal).then_some(expr)
    })
}

/// # Type constructor tactic
///
/// Attempts different type constructors for enums and structs in scope
///
/// Updates lookup by new types reached and returns iterator that yields
/// elements that unify with `goal`.
///
/// # Arguments
/// * `ctx` - Context for the term search
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
pub(super) fn type_constructor<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    fn variant_helper(
        db: &dyn HirDatabase,
        lookup: &mut LookupTable,
        parent_enum: Enum,
        variant: Variant,
        config: &TermSearchConfig,
    ) -> Vec<(Type, Vec<Expr>)> {
        // Ignore unstable
        if variant.is_unstable(db) {
            return Vec::new();
        }

        let generics = GenericDef::from(variant.parent_enum(db));
        let Some(type_params) = generics
            .type_or_const_params(db)
            .into_iter()
            .map(|it| it.as_type_param(db))
            .collect::<Option<Vec<TypeParam>>>()
        else {
            // Ignore enums with const generics
            return Vec::new();
        };

        // We currently do not check lifetime bounds so ignore all types that have something to do
        // with them
        if !generics.lifetime_params(db).is_empty() {
            return Vec::new();
        }

        // Only account for stable type parameters for now, unstable params can be default
        // tho, for example in `Box<T, #[unstable] A: Allocator>`
        if type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none()) {
            return Vec::new();
        }

        let non_default_type_params_len =
            type_params.iter().filter(|it| it.default(db).is_none()).count();

        let enum_ty_shallow = Adt::from(parent_enum).ty(db);
        let generic_params = lookup
            .types_wishlist()
            .clone()
            .into_iter()
            .filter(|ty| ty.could_unify_with(db, &enum_ty_shallow))
            .map(|it| it.type_arguments().collect::<Vec<Type>>())
            .chain((non_default_type_params_len == 0).then_some(Vec::new()));

        generic_params
            .filter_map(move |generics| {
                // Insert default type params
                let mut g = generics.into_iter();
                let generics: Vec<_> = type_params
                    .iter()
                    .map(|it| it.default(db).or_else(|| g.next()))
                    .collect::<Option<_>>()?;

                let enum_ty = Adt::from(parent_enum).ty_with_args(db, generics.iter().cloned());

                // Ignore types that have something to do with lifetimes
                if config.enable_borrowcheck && enum_ty.contains_reference(db) {
                    return None;
                }

                // Early exit if some param cannot be filled from lookup
                let param_exprs: Vec<Vec<Expr>> = variant
                    .fields(db)
                    .into_iter()
                    .map(|field| lookup.find(db, &field.ty_with_args(db, generics.iter().cloned())))
                    .collect::<Option<_>>()?;

                // Note that we need special case for 0 param constructors because of multi cartesian
                // product
                let variant_exprs: Vec<Expr> = if param_exprs.is_empty() {
                    vec![Expr::Variant { variant, generics: generics.clone(), params: Vec::new() }]
                } else {
                    param_exprs
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|params| Expr::Variant { variant, generics: generics.clone(), params })
                        .collect()
                };
                lookup.insert(enum_ty.clone(), variant_exprs.iter().cloned());

                Some((enum_ty, variant_exprs))
            })
            .collect()
    }
    defs.iter()
        .filter_map(move |def| match def {
            ScopeDef::ModuleDef(ModuleDef::Variant(it)) => {
                let variant_exprs =
                    variant_helper(db, lookup, it.parent_enum(db), *it, &ctx.config);
                if variant_exprs.is_empty() {
                    return None;
                }
                if GenericDef::from(it.parent_enum(db))
                    .type_or_const_params(db)
                    .into_iter()
                    .filter_map(|it| it.as_type_param(db))
                    .all(|it| it.default(db).is_some())
                {
                    lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Variant(*it)));
                }
                Some(variant_exprs)
            }
            ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Enum(enum_))) => {
                let exprs: Vec<(Type, Vec<Expr>)> = enum_
                    .variants(db)
                    .into_iter()
                    .flat_map(|it| variant_helper(db, lookup, *enum_, it, &ctx.config))
                    .collect();

                if exprs.is_empty() {
                    return None;
                }

                if GenericDef::from(*enum_)
                    .type_or_const_params(db)
                    .into_iter()
                    .filter_map(|it| it.as_type_param(db))
                    .all(|it| it.default(db).is_some())
                {
                    lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Enum(*enum_))));
                }

                Some(exprs)
            }
            ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Struct(it))) => {
                // Ignore unstable and not visible
                if it.is_unstable(db) || !it.is_visible_from(db, module) {
                    return None;
                }

                let generics = GenericDef::from(*it);

                // Ignore const params for now
                let type_params = generics
                    .type_or_const_params(db)
                    .into_iter()
                    .map(|it| it.as_type_param(db))
                    .collect::<Option<Vec<TypeParam>>>()?;

                // We currently do not check lifetime bounds so ignore all types that have something to do
                // with them
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

                let struct_ty_shallow = Adt::from(*it).ty(db);
                let generic_params = lookup
                    .types_wishlist()
                    .clone()
                    .into_iter()
                    .filter(|ty| ty.could_unify_with(db, &struct_ty_shallow))
                    .map(|it| it.type_arguments().collect::<Vec<Type>>())
                    .chain((non_default_type_params_len == 0).then_some(Vec::new()));

                let exprs = generic_params
                    .filter_map(|generics| {
                        // Insert default type params
                        let mut g = generics.into_iter();
                        let generics: Vec<_> = type_params
                            .iter()
                            .map(|it| it.default(db).or_else(|| g.next()))
                            .collect::<Option<_>>()?;

                        let struct_ty = Adt::from(*it).ty_with_args(db, generics.iter().cloned());

                        // Ignore types that have something to do with lifetimes
                        if ctx.config.enable_borrowcheck && struct_ty.contains_reference(db) {
                            return None;
                        }
                        let fields = it.fields(db);
                        // Check if all fields are visible, otherwise we cannot fill them
                        if fields.iter().any(|it| !it.is_visible_from(db, module)) {
                            return None;
                        }

                        // Early exit if some param cannot be filled from lookup
                        let param_exprs: Vec<Vec<Expr>> = fields
                            .into_iter()
                            .map(|field| lookup.find(db, &field.ty(db)))
                            .collect::<Option<_>>()?;

                        // Note that we need special case for 0 param constructors because of multi cartesian
                        // product
                        let struct_exprs: Vec<Expr> = if param_exprs.is_empty() {
                            vec![Expr::Struct { strukt: *it, generics, params: Vec::new() }]
                        } else {
                            param_exprs
                                .into_iter()
                                .multi_cartesian_product()
                                .map(|params| Expr::Struct {
                                    strukt: *it,
                                    generics: generics.clone(),
                                    params,
                                })
                                .collect()
                        };

                        if non_default_type_params_len == 0 {
                            // Fulfilled only if there are no generic parameters
                            lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Adt(
                                Adt::Struct(*it),
                            )));
                        }
                        lookup.insert(struct_ty.clone(), struct_exprs.iter().cloned());

                        Some((struct_ty, struct_exprs))
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
pub(super) fn free_function<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
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
                            || it.is_unsafe_to_call(db)
                            || it.is_unstable(db)
                            || ctx.config.enable_borrowcheck && ret_ty.contains_reference(db)
                            || ret_ty.is_raw_ptr()
                        {
                            return None;
                        }

                        // Early exit if some param cannot be filled from lookup
                        let param_exprs: Vec<Vec<Expr>> = it
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
                        let fn_exprs: Vec<Expr> = if param_exprs.is_empty() {
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

                        lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Function(*it)));
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
/// Attempts to to call methods on types from lookup table.
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
pub(super) fn impl_method<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .new_types(NewTypesKey::ImplMethod)
        .into_iter()
        .flat_map(|ty| {
            Impl::all_for_type(db, ty.clone()).into_iter().map(move |imp| (ty.clone(), imp))
        })
        .flat_map(|(ty, imp)| imp.items(db).into_iter().map(move |item| (imp, ty.clone(), item)))
        .filter_map(|(imp, ty, it)| match it {
            AssocItem::Function(f) => Some((imp, ty, f)),
            _ => None,
        })
        .filter_map(move |(imp, ty, it)| {
            let fn_generics = GenericDef::from(it);
            let imp_generics = GenericDef::from(imp);

            // Ignore const params for now
            let imp_type_params = imp_generics
                .type_or_const_params(db)
                .into_iter()
                .map(|it| it.as_type_param(db))
                .collect::<Option<Vec<TypeParam>>>()?;

            // Ignore const params for now
            let fn_type_params = fn_generics
                .type_or_const_params(db)
                .into_iter()
                .map(|it| it.as_type_param(db))
                .collect::<Option<Vec<TypeParam>>>()?;

            // Ignore all functions that have something to do with lifetimes as we don't check them
            if !fn_generics.lifetime_params(db).is_empty() {
                return None;
            }

            // Ignore functions without self param
            if !it.has_self_param(db) {
                return None;
            }

            // Filter out private and unsafe functions
            if !it.is_visible_from(db, module) || it.is_unsafe_to_call(db) || it.is_unstable(db) {
                return None;
            }

            // Only account for stable type parameters for now, unstable params can be default
            // tho, for example in `Box<T, #[unstable] A: Allocator>`
            if imp_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
                || fn_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
            {
                return None;
            }

            // Double check that we have fully known type
            if ty.type_arguments().any(|it| it.contains_unknown()) {
                return None;
            }

            let non_default_fn_type_params_len =
                fn_type_params.iter().filter(|it| it.default(db).is_none()).count();

            // Ignore functions with generics for now as they kill the performance
            // Also checking bounds for generics is problematic
            if non_default_fn_type_params_len > 0 {
                return None;
            }

            let generic_params = lookup
                .iter_types()
                .collect::<Vec<_>>() // Force take ownership
                .into_iter()
                .permutations(non_default_fn_type_params_len);

            let exprs: Vec<_> = generic_params
                .filter_map(|generics| {
                    // Insert default type params
                    let mut g = generics.into_iter();
                    let generics: Vec<_> = ty
                        .type_arguments()
                        .map(Some)
                        .chain(fn_type_params.iter().map(|it| match it.default(db) {
                            Some(ty) => Some(ty),
                            None => {
                                let generic = g.next().expect("Missing type param");
                                // Filter out generics that do not unify due to trait bounds
                                it.ty(db).could_unify_with(db, &generic).then_some(generic)
                            }
                        }))
                        .collect::<Option<_>>()?;

                    let ret_ty = it.ret_type_with_args(
                        db,
                        ty.type_arguments().chain(generics.iter().cloned()),
                    );
                    // Filter out functions that return references
                    if ctx.config.enable_borrowcheck && ret_ty.contains_reference(db)
                        || ret_ty.is_raw_ptr()
                    {
                        return None;
                    }

                    // Ignore functions that do not change the type
                    if ty.could_unify_with_deeply(db, &ret_ty) {
                        return None;
                    }

                    let self_ty = it
                        .self_param(db)
                        .expect("No self param")
                        .ty_with_args(db, ty.type_arguments().chain(generics.iter().cloned()));

                    // Ignore functions that have different self type
                    if !self_ty.autoderef(db).any(|s_ty| ty == s_ty) {
                        return None;
                    }

                    let target_type_exprs = lookup.find(db, &ty).expect("Type not in lookup");

                    // Early exit if some param cannot be filled from lookup
                    let param_exprs: Vec<Vec<Expr>> = it
                        .params_without_self_with_args(
                            db,
                            ty.type_arguments().chain(generics.iter().cloned()),
                        )
                        .into_iter()
                        .map(|field| lookup.find_autoref(db, field.ty()))
                        .collect::<Option<_>>()?;

                    let fn_exprs: Vec<Expr> = std::iter::once(target_type_exprs)
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

                    lookup.insert(ret_ty.clone(), fn_exprs.iter().cloned());
                    Some((ret_ty, fn_exprs))
                })
                .collect();
            Some(exprs)
        })
        .flatten()
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
pub(super) fn struct_projection<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .new_types(NewTypesKey::StructProjection)
        .into_iter()
        .map(|ty| (ty.clone(), lookup.find(db, &ty).expect("Expr not in lookup")))
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
pub(super) fn famous_types<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    [
        Expr::FamousType { ty: Type::new(db, module.id, TyBuilder::bool()), value: "true" },
        Expr::FamousType { ty: Type::new(db, module.id, TyBuilder::bool()), value: "false" },
        Expr::FamousType { ty: Type::new(db, module.id, TyBuilder::unit()), value: "()" },
    ]
    .into_iter()
    .map(|exprs| {
        lookup.insert(exprs.ty(db), std::iter::once(exprs.clone()));
        exprs
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
pub(super) fn impl_static_method<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .types_wishlist()
        .clone()
        .into_iter()
        .chain(iter::once(ctx.goal.clone()))
        .flat_map(|ty| {
            Impl::all_for_type(db, ty.clone()).into_iter().map(move |imp| (ty.clone(), imp))
        })
        .filter(|(_, imp)| !imp.is_unsafe(db))
        .flat_map(|(ty, imp)| imp.items(db).into_iter().map(move |item| (imp, ty.clone(), item)))
        .filter_map(|(imp, ty, it)| match it {
            AssocItem::Function(f) => Some((imp, ty, f)),
            _ => None,
        })
        .filter_map(move |(imp, ty, it)| {
            let fn_generics = GenericDef::from(it);
            let imp_generics = GenericDef::from(imp);

            // Ignore const params for now
            let imp_type_params = imp_generics
                .type_or_const_params(db)
                .into_iter()
                .map(|it| it.as_type_param(db))
                .collect::<Option<Vec<TypeParam>>>()?;

            // Ignore const params for now
            let fn_type_params = fn_generics
                .type_or_const_params(db)
                .into_iter()
                .map(|it| it.as_type_param(db))
                .collect::<Option<Vec<TypeParam>>>()?;

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
            if !it.is_visible_from(db, module) || it.is_unsafe_to_call(db) || it.is_unstable(db) {
                return None;
            }

            // Only account for stable type parameters for now, unstable params can be default
            // tho, for example in `Box<T, #[unstable] A: Allocator>`
            if imp_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
                || fn_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
            {
                return None;
            }

            // Double check that we have fully known type
            if ty.type_arguments().any(|it| it.contains_unknown()) {
                return None;
            }

            let non_default_fn_type_params_len =
                fn_type_params.iter().filter(|it| it.default(db).is_none()).count();

            // Ignore functions with generics for now as they kill the performance
            // Also checking bounds for generics is problematic
            if non_default_fn_type_params_len > 0 {
                return None;
            }

            let generic_params = lookup
                .iter_types()
                .collect::<Vec<_>>() // Force take ownership
                .into_iter()
                .permutations(non_default_fn_type_params_len);

            let exprs: Vec<_> = generic_params
                .filter_map(|generics| {
                    // Insert default type params
                    let mut g = generics.into_iter();
                    let generics: Vec<_> = ty
                        .type_arguments()
                        .map(Some)
                        .chain(fn_type_params.iter().map(|it| match it.default(db) {
                            Some(ty) => Some(ty),
                            None => {
                                let generic = g.next().expect("Missing type param");
                                it.trait_bounds(db)
                                    .into_iter()
                                    .all(|bound| generic.impls_trait(db, bound, &[]));
                                // Filter out generics that do not unify due to trait bounds
                                it.ty(db).could_unify_with(db, &generic).then_some(generic)
                            }
                        }))
                        .collect::<Option<_>>()?;

                    let ret_ty = it.ret_type_with_args(
                        db,
                        ty.type_arguments().chain(generics.iter().cloned()),
                    );
                    // Filter out functions that return references
                    if ctx.config.enable_borrowcheck && ret_ty.contains_reference(db)
                        || ret_ty.is_raw_ptr()
                    {
                        return None;
                    }

                    // Ignore functions that do not change the type
                    // if ty.could_unify_with_deeply(db, &ret_ty) {
                    //     return None;
                    // }

                    // Early exit if some param cannot be filled from lookup
                    let param_exprs: Vec<Vec<Expr>> = it
                        .params_without_self_with_args(
                            db,
                            ty.type_arguments().chain(generics.iter().cloned()),
                        )
                        .into_iter()
                        .map(|field| lookup.find_autoref(db, field.ty()))
                        .collect::<Option<_>>()?;

                    // Note that we need special case for 0 param constructors because of multi cartesian
                    // product
                    let fn_exprs: Vec<Expr> = if param_exprs.is_empty() {
                        vec![Expr::Function { func: it, generics, params: Vec::new() }]
                    } else {
                        param_exprs
                            .into_iter()
                            .multi_cartesian_product()
                            .map(|params| Expr::Function {
                                func: it,
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
        })
        .flatten()
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
pub(super) fn make_tuple<'a, DB: HirDatabase>(
    ctx: &'a TermSearchCtx<'a, DB>,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
) -> impl Iterator<Item = Expr> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();

    lookup
        .types_wishlist()
        .clone()
        .into_iter()
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
            let param_exprs: Vec<Vec<Expr>> =
                ty.type_arguments().map(|field| lookup.find(db, &field)).collect::<Option<_>>()?;

            let exprs: Vec<Expr> = param_exprs
                .into_iter()
                .multi_cartesian_product()
                .map(|params| {
                    let tys: Vec<Type> = params.iter().map(|it| it.ty(db)).collect();
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
