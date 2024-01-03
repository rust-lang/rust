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
    Variant,
};

use crate::term_search::{TermSearchConfig, TypeTree};

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
) -> impl Iterator<Item = TypeTree> + 'a {
    let db = ctx.sema.db;
    defs.iter().filter_map(|def| {
        let tt = match def {
            ScopeDef::ModuleDef(ModuleDef::Const(it)) => Some(TypeTree::Const(*it)),
            ScopeDef::ModuleDef(ModuleDef::Static(it)) => Some(TypeTree::Static(*it)),
            ScopeDef::GenericParam(GenericParam::ConstParam(it)) => Some(TypeTree::ConstParam(*it)),
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

                Some(TypeTree::Local(*it))
            }
            _ => None,
        }?;

        lookup.mark_exhausted(*def);

        let ty = tt.ty(db);
        lookup.insert(ty.clone(), std::iter::once(tt.clone()));

        // Don't suggest local references as they are not valid for return
        if matches!(tt, TypeTree::Local(_)) && ty.contains_reference(db) {
            return None;
        }

        ty.could_unify_with_deeply(db, &ctx.goal).then(|| tt)
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
) -> impl Iterator<Item = TypeTree> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    fn variant_helper(
        db: &dyn HirDatabase,
        lookup: &mut LookupTable,
        parent_enum: Enum,
        variant: Variant,
        goal: &Type,
        config: &TermSearchConfig,
    ) -> Vec<(Type, Vec<TypeTree>)> {
        let generics = GenericDef::from(variant.parent_enum(db));

        // Ignore unstable variants
        if variant.is_unstable(db) {
            return Vec::new();
        }

        // Ignore enums with const generics
        if !generics.const_params(db).is_empty() {
            return Vec::new();
        }

        // We currently do not check lifetime bounds so ignore all types that have something to do
        // with them
        if !generics.lifetime_params(db).is_empty() {
            return Vec::new();
        }

        // Only account for stable type parameters for now
        let type_params = generics.type_params(db);

        // Only account for stable type parameters for now, unstable params can be default
        // tho, for example in `Box<T, #[unstable] A: Allocator>`
        if type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none()) {
            return Vec::new();
        }

        let non_default_type_params_len =
            type_params.iter().filter(|it| it.default(db).is_none()).count();

        let generic_params = lookup
            .iter_types()
            .collect::<Vec<_>>() // Force take ownership
            .into_iter()
            .permutations(non_default_type_params_len);

        generic_params
            .filter_map(move |generics| {
                // Insert default type params
                let mut g = generics.into_iter();
                let generics: Vec<_> = type_params
                    .iter()
                    .map(|it| match it.default(db) {
                        Some(ty) => ty,
                        None => g.next().expect("Missing type param"),
                    })
                    .collect();

                let enum_ty = parent_enum.ty_with_generics(db, generics.iter().cloned());

                // Allow types with generics only if they take us straight to goal for
                // performance reasons
                if !generics.is_empty() && !enum_ty.could_unify_with_deeply(db, goal) {
                    return None;
                }

                // Ignore types that have something to do with lifetimes
                if config.enable_borrowcheck && enum_ty.contains_reference(db) {
                    return None;
                }

                // Early exit if some param cannot be filled from lookup
                let param_trees: Vec<Vec<TypeTree>> = variant
                    .fields(db)
                    .into_iter()
                    .map(|field| {
                        lookup.find(db, &field.ty_with_generics(db, generics.iter().cloned()))
                    })
                    .collect::<Option<_>>()?;

                // Note that we need special case for 0 param constructors because of multi cartesian
                // product
                let variant_trees: Vec<TypeTree> = if param_trees.is_empty() {
                    vec![TypeTree::Variant {
                        variant,
                        generics: generics.clone(),
                        params: Vec::new(),
                    }]
                } else {
                    param_trees
                        .into_iter()
                        .multi_cartesian_product()
                        .map(|params| TypeTree::Variant {
                            variant,
                            generics: generics.clone(),
                            params,
                        })
                        .collect()
                };
                lookup.insert(enum_ty.clone(), variant_trees.iter().cloned());

                Some((enum_ty, variant_trees))
            })
            .collect()
    }
    defs.iter()
        .filter_map(move |def| match def {
            ScopeDef::ModuleDef(ModuleDef::Variant(it)) => {
                let variant_trees =
                    variant_helper(db, lookup, it.parent_enum(db), *it, &ctx.goal, &ctx.config);
                if variant_trees.is_empty() {
                    return None;
                }
                lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Variant(*it)));
                Some(variant_trees)
            }
            ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Enum(enum_))) => {
                let trees: Vec<(Type, Vec<TypeTree>)> = enum_
                    .variants(db)
                    .into_iter()
                    .flat_map(|it| {
                        variant_helper(db, lookup, enum_.clone(), it, &ctx.goal, &ctx.config)
                    })
                    .collect();

                if !trees.is_empty() {
                    lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Enum(*enum_))));
                }

                Some(trees)
            }
            ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Struct(it))) => {
                // Ignore unstable and not visible
                if it.is_unstable(db) || !it.is_visible_from(db, module) {
                    return None;
                }

                let generics = GenericDef::from(*it);

                // Ignore enums with const generics
                if !generics.const_params(db).is_empty() {
                    return None;
                }

                // We currently do not check lifetime bounds so ignore all types that have something to do
                // with them
                if !generics.lifetime_params(db).is_empty() {
                    return None;
                }

                let type_params = generics.type_params(db);

                // Only account for stable type parameters for now, unstable params can be default
                // tho, for example in `Box<T, #[unstable] A: Allocator>`
                if type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none()) {
                    return None;
                }

                let non_default_type_params_len =
                    type_params.iter().filter(|it| it.default(db).is_none()).count();

                let generic_params = lookup
                    .iter_types()
                    .collect::<Vec<_>>() // Force take ownership
                    .into_iter()
                    .permutations(non_default_type_params_len);

                let trees = generic_params
                    .filter_map(|generics| {
                        // Insert default type params
                        let mut g = generics.into_iter();
                        let generics: Vec<_> = type_params
                            .iter()
                            .map(|it| match it.default(db) {
                                Some(ty) => ty,
                                None => g.next().expect("Missing type param"),
                            })
                            .collect();
                        let struct_ty = it.ty_with_generics(db, generics.iter().cloned());

                        // Allow types with generics only if they take us straight to goal for
                        // performance reasons
                        if non_default_type_params_len != 0
                            && struct_ty.could_unify_with_deeply(db, &ctx.goal)
                        {
                            return None;
                        }

                        // Ignore types that have something to do with lifetimes
                        if ctx.config.enable_borrowcheck && struct_ty.contains_reference(db) {
                            return None;
                        }
                        let fileds = it.fields(db);
                        // Check if all fields are visible, otherwise we cannot fill them
                        if fileds.iter().any(|it| !it.is_visible_from(db, module)) {
                            return None;
                        }

                        // Early exit if some param cannot be filled from lookup
                        let param_trees: Vec<Vec<TypeTree>> = fileds
                            .into_iter()
                            .map(|field| lookup.find(db, &field.ty(db)))
                            .collect::<Option<_>>()?;

                        // Note that we need special case for 0 param constructors because of multi cartesian
                        // product
                        let struct_trees: Vec<TypeTree> = if param_trees.is_empty() {
                            vec![TypeTree::Struct { strukt: *it, generics, params: Vec::new() }]
                        } else {
                            param_trees
                                .into_iter()
                                .multi_cartesian_product()
                                .map(|params| TypeTree::Struct {
                                    strukt: *it,
                                    generics: generics.clone(),
                                    params,
                                })
                                .collect()
                        };

                        lookup
                            .mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Struct(*it))));
                        lookup.insert(struct_ty.clone(), struct_trees.iter().cloned());

                        Some((struct_ty, struct_trees))
                    })
                    .collect();
                Some(trees)
            }
            _ => None,
        })
        .flatten()
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, &ctx.goal).then(|| trees))
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
) -> impl Iterator<Item = TypeTree> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    defs.iter()
        .filter_map(move |def| match def {
            ScopeDef::ModuleDef(ModuleDef::Function(it)) => {
                let generics = GenericDef::from(*it);

                // Skip functions that require const generics
                if !generics.const_params(db).is_empty() {
                    return None;
                }

                // Ignore lifetimes as we do not check them
                if !generics.lifetime_params(db).is_empty() {
                    return None;
                }

                let type_params = generics.type_params(db);

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

                let trees: Vec<_> = generic_params
                    .filter_map(|generics| {
                        // Insert default type params
                        let mut g = generics.into_iter();
                        let generics: Vec<_> = type_params
                            .iter()
                            .map(|it| match it.default(db) {
                                Some(ty) => ty,
                                None => g.next().expect("Missing type param"),
                            })
                            .collect();

                        let ret_ty = it.ret_type_with_generics(db, generics.iter().cloned());
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
                        let param_trees: Vec<Vec<TypeTree>> = it
                            .params_without_self_with_generics(db, generics.iter().cloned())
                            .into_iter()
                            .map(|field| {
                                let ty = field.ty();
                                match ty.is_mutable_reference() {
                                    true => None,
                                    false => lookup.find_autoref(db, &ty),
                                }
                            })
                            .collect::<Option<_>>()?;

                        // Note that we need special case for 0 param constructors because of multi cartesian
                        // product
                        let fn_trees: Vec<TypeTree> = if param_trees.is_empty() {
                            vec![TypeTree::Function { func: *it, generics, params: Vec::new() }]
                        } else {
                            param_trees
                                .into_iter()
                                .multi_cartesian_product()
                                .map(|params| TypeTree::Function {
                                    func: *it,
                                    generics: generics.clone(),

                                    params,
                                })
                                .collect()
                        };

                        lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Function(*it)));
                        lookup.insert(ret_ty.clone(), fn_trees.iter().cloned());
                        Some((ret_ty, fn_trees))
                    })
                    .collect();
                Some(trees)
            }
            _ => None,
        })
        .flatten()
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, &ctx.goal).then(|| trees))
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
) -> impl Iterator<Item = TypeTree> + 'a {
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

            // Ignore impl if it has const type arguments
            if !fn_generics.const_params(db).is_empty() || !imp_generics.const_params(db).is_empty()
            {
                return None;
            }

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

            let imp_type_params = imp_generics.type_params(db);
            let fn_type_params = fn_generics.type_params(db);

            // Only account for stable type parameters for now, unstable params can be default
            // tho, for example in `Box<T, #[unstable] A: Allocator>`
            if imp_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
                || fn_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
            {
                return None;
            }

            let non_default_type_params_len = imp_type_params
                .iter()
                .chain(fn_type_params.iter())
                .filter(|it| it.default(db).is_none())
                .count();

            // Ignore bigger number of generics for now as they kill the performance
            if non_default_type_params_len > 0 {
                return None;
            }

            let generic_params = lookup
                .iter_types()
                .collect::<Vec<_>>() // Force take ownership
                .into_iter()
                .permutations(non_default_type_params_len);

            let trees: Vec<_> = generic_params
                .filter_map(|generics| {
                    // Insert default type params
                    let mut g = generics.into_iter();
                    let generics: Vec<_> = imp_type_params
                        .iter()
                        .chain(fn_type_params.iter())
                        .map(|it| match it.default(db) {
                            Some(ty) => ty,
                            None => g.next().expect("Missing type param"),
                        })
                        .collect();

                    let ret_ty = it.ret_type_with_generics(
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
                        .ty_with_generics(db, ty.type_arguments().chain(generics.iter().cloned()));

                    // Ignore functions that have different self type
                    if !self_ty.autoderef(db).any(|s_ty| ty == s_ty) {
                        return None;
                    }

                    let target_type_trees = lookup.find(db, &ty).expect("Type not in lookup");

                    // Early exit if some param cannot be filled from lookup
                    let param_trees: Vec<Vec<TypeTree>> = it
                        .params_without_self_with_generics(
                            db,
                            ty.type_arguments().chain(generics.iter().cloned()),
                        )
                        .into_iter()
                        .map(|field| lookup.find_autoref(db, &field.ty()))
                        .collect::<Option<_>>()?;

                    let fn_trees: Vec<TypeTree> = std::iter::once(target_type_trees)
                        .chain(param_trees.into_iter())
                        .multi_cartesian_product()
                        .map(|params| TypeTree::Function { func: it, generics: Vec::new(), params })
                        .collect();

                    lookup.insert(ret_ty.clone(), fn_trees.iter().cloned());
                    Some((ret_ty, fn_trees))
                })
                .collect();
            Some(trees)
        })
        .flatten()
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, &ctx.goal).then(|| trees))
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
) -> impl Iterator<Item = TypeTree> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .new_types(NewTypesKey::StructProjection)
        .into_iter()
        .map(|ty| (ty.clone(), lookup.find(db, &ty).expect("TypeTree not in lookup")))
        .flat_map(move |(ty, targets)| {
            ty.fields(db).into_iter().filter_map(move |(field, filed_ty)| {
                if !field.is_visible_from(db, module) {
                    return None;
                }
                let trees = targets
                    .clone()
                    .into_iter()
                    .map(move |target| TypeTree::Field { field, type_tree: Box::new(target) });
                Some((filed_ty, trees))
            })
        })
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, &ctx.goal).then(|| trees))
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
) -> impl Iterator<Item = TypeTree> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    [
        TypeTree::FamousType { ty: Type::new(db, module.id, TyBuilder::bool()), value: "true" },
        TypeTree::FamousType { ty: Type::new(db, module.id, TyBuilder::bool()), value: "false" },
        TypeTree::FamousType { ty: Type::new(db, module.id, TyBuilder::unit()), value: "()" },
    ]
    .into_iter()
    .map(|tt| {
        lookup.insert(tt.ty(db), std::iter::once(tt.clone()));
        tt
    })
    .filter(|tt| tt.ty(db).could_unify_with_deeply(db, &ctx.goal))
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
) -> impl Iterator<Item = TypeTree> + 'a {
    let db = ctx.sema.db;
    let module = ctx.scope.module();
    lookup
        .take_types_wishlist()
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

            // Ignore impl if it has const type arguments
            if !fn_generics.const_params(db).is_empty() || !imp_generics.const_params(db).is_empty()
            {
                return None;
            }

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

            let imp_type_params = imp_generics.type_params(db);
            let fn_type_params = fn_generics.type_params(db);

            // Only account for stable type parameters for now, unstable params can be default
            // tho, for example in `Box<T, #[unstable] A: Allocator>`
            if imp_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
                || fn_type_params.iter().any(|it| it.is_unstable(db) && it.default(db).is_none())
            {
                return None;
            }

            let non_default_type_params_len = imp_type_params
                .iter()
                .chain(fn_type_params.iter())
                .filter(|it| it.default(db).is_none())
                .count();

            // Ignore bigger number of generics for now as they kill the performance
            if non_default_type_params_len > 0 {
                return None;
            }

            let generic_params = lookup
                .iter_types()
                .collect::<Vec<_>>() // Force take ownership
                .into_iter()
                .permutations(non_default_type_params_len);

            let trees: Vec<_> = generic_params
                .filter_map(|generics| {
                    // Insert default type params
                    let mut g = generics.into_iter();
                    let generics: Vec<_> = imp_type_params
                        .iter()
                        .chain(fn_type_params.iter())
                        .map(|it| match it.default(db) {
                            Some(ty) => ty,
                            None => g.next().expect("Missing type param"),
                        })
                        .collect();

                    let ret_ty = it.ret_type_with_generics(
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
                    let param_trees: Vec<Vec<TypeTree>> = it
                        .params_without_self_with_generics(
                            db,
                            ty.type_arguments().chain(generics.iter().cloned()),
                        )
                        .into_iter()
                        .map(|field| lookup.find_autoref(db, &field.ty()))
                        .collect::<Option<_>>()?;

                    // Note that we need special case for 0 param constructors because of multi cartesian
                    // product
                    let fn_trees: Vec<TypeTree> = if param_trees.is_empty() {
                        vec![TypeTree::Function { func: it, generics, params: Vec::new() }]
                    } else {
                        param_trees
                            .into_iter()
                            .multi_cartesian_product()
                            .map(|params| TypeTree::Function {
                                func: it,
                                generics: generics.clone(),

                                params,
                            })
                            .collect()
                    };

                    lookup.insert(ret_ty.clone(), fn_trees.iter().cloned());
                    Some((ret_ty, fn_trees))
                })
                .collect();
            Some(trees)
        })
        .flatten()
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, &ctx.goal).then(|| trees))
        .flatten()
}
