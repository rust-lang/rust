//! Tactics for term search

use hir_def::generics::TypeOrConstParamData;
use hir_ty::db::HirDatabase;
use hir_ty::mir::BorrowKind;
use hir_ty::TyBuilder;
use itertools::Itertools;
use rustc_hash::FxHashSet;

use crate::{
    Adt, AssocItem, Enum, GenericParam, HasVisibility, Impl, Module, ModuleDef, ScopeDef, Type,
    Variant,
};

use crate::term_search::TypeTree;

use super::{LookupTable, NewTypesKey, MAX_VARIATIONS};

/// Trivial tactic
///
/// Attempts to fulfill the goal by trying items in scope
/// Also works as a starting point to move all items in scope to lookup table
pub(super) fn trivial<'a>(
    db: &'a dyn HirDatabase,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
    goal: &'a Type,
) -> impl Iterator<Item = TypeTree> + 'a {
    defs.iter().filter_map(|def| {
        let tt = match def {
            ScopeDef::ModuleDef(ModuleDef::Const(it)) => Some(TypeTree::Const(*it)),
            ScopeDef::ModuleDef(ModuleDef::Static(it)) => Some(TypeTree::Static(*it)),
            ScopeDef::GenericParam(GenericParam::ConstParam(it)) => Some(TypeTree::ConstParam(*it)),
            ScopeDef::Local(it) => {
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

                Some(TypeTree::Local(*it))
            }
            _ => None,
        }?;

        lookup.mark_exhausted(*def);

        let ty = tt.ty(db);
        lookup.insert(ty.clone(), std::iter::once(tt.clone()));

        // Don't suggest local references as they are not valid for return
        if matches!(tt, TypeTree::Local(_)) && ty.is_reference() {
            return None;
        }

        ty.could_unify_with_deeply(db, goal).then(|| tt)
    })
}

/// Type constructor tactic
///
/// Attempts different type constructors for enums and structs in scope
///
/// # Arguments
/// * `db` - HIR database
/// * `module` - Module where the term search target location
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `goal` - Term search target type
pub(super) fn type_constructor<'a>(
    db: &'a dyn HirDatabase,
    module: &'a Module,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
    goal: &'a Type,
) -> impl Iterator<Item = TypeTree> + 'a {
    fn variant_helper(
        db: &dyn HirDatabase,
        lookup: &mut LookupTable,
        parent_enum: Enum,
        variant: Variant,
        goal: &Type,
    ) -> Vec<(Type, Vec<TypeTree>)> {
        let generics = db.generic_params(variant.parent_enum(db).id.into());

        // Ignore enums with const generics
        if generics
            .type_or_consts
            .values()
            .any(|it| matches!(it, TypeOrConstParamData::ConstParamData(_)))
        {
            return Vec::new();
        }

        // We currently do not check lifetime bounds so ignore all types that have something to do
        // with them
        if !generics.lifetimes.is_empty() {
            return Vec::new();
        }

        let generic_params = lookup
            .iter_types()
            .collect::<Vec<_>>() // Force take ownership
            .into_iter()
            .permutations(generics.type_or_consts.len());

        generic_params
            .filter_map(|generics| {
                let enum_ty = parent_enum.ty_with_generics(db, generics.iter().cloned());

                if !generics.is_empty() && !enum_ty.could_unify_with_deeply(db, goal) {
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
                        .take(MAX_VARIATIONS)
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
        .filter_map(|def| match def {
            ScopeDef::ModuleDef(ModuleDef::Variant(it)) => {
                let variant_trees = variant_helper(db, lookup, it.parent_enum(db), *it, goal);
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
                    .flat_map(|it| variant_helper(db, lookup, enum_.clone(), it, goal))
                    .collect();

                if !trees.is_empty() {
                    lookup.mark_fulfilled(ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Enum(*enum_))));
                }

                Some(trees)
            }
            ScopeDef::ModuleDef(ModuleDef::Adt(Adt::Struct(it))) => {
                let generics = db.generic_params(it.id.into());

                // Ignore enums with const generics
                if generics
                    .type_or_consts
                    .values()
                    .any(|it| matches!(it, TypeOrConstParamData::ConstParamData(_)))
                {
                    return None;
                }

                // We currently do not check lifetime bounds so ignore all types that have something to do
                // with them
                if !generics.lifetimes.is_empty() {
                    return None;
                }

                let generic_params = lookup
                    .iter_types()
                    .collect::<Vec<_>>() // Force take ownership
                    .into_iter()
                    .permutations(generics.type_or_consts.len());

                let trees = generic_params
                    .filter_map(|generics| {
                        let struct_ty = it.ty_with_generics(db, generics.iter().cloned());
                        if !generics.is_empty() && !struct_ty.could_unify_with_deeply(db, goal) {
                            return None;
                        }
                        let fileds = it.fields(db);
                        // Check if all fields are visible, otherwise we cannot fill them
                        if fileds.iter().any(|it| !it.is_visible_from(db, *module)) {
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
                                .take(MAX_VARIATIONS)
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
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, goal).then(|| trees))
        .flatten()
}

/// Free function tactic
///
/// Attempts to call different functions in scope with parameters from lookup table
///
/// # Arguments
/// * `db` - HIR database
/// * `module` - Module where the term search target location
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `goal` - Term search target type
pub(super) fn free_function<'a>(
    db: &'a dyn HirDatabase,
    module: &'a Module,
    defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
    goal: &'a Type,
) -> impl Iterator<Item = TypeTree> + 'a {
    defs.iter()
        .filter_map(|def| match def {
            ScopeDef::ModuleDef(ModuleDef::Function(it)) => {
                let generics = db.generic_params(it.id.into());

                // Skip functions that require const generics
                if generics
                    .type_or_consts
                    .values()
                    .any(|it| matches!(it, TypeOrConstParamData::ConstParamData(_)))
                {
                    return None;
                }

                // Ignore bigger number of generics for now as they kill the performance
                // Ignore lifetimes as we do not check them
                if generics.type_or_consts.len() > 0 || !generics.lifetimes.is_empty() {
                    return None;
                }

                let generic_params = lookup
                    .iter_types()
                    .collect::<Vec<_>>() // Force take ownership
                    .into_iter()
                    .permutations(generics.type_or_consts.len());

                let trees: Vec<_> = generic_params
                    .filter_map(|generics| {
                        let ret_ty = it.ret_type_with_generics(db, generics.iter().cloned());
                        // Filter out private and unsafe functions
                        if !it.is_visible_from(db, *module)
                            || it.is_unsafe_to_call(db)
                            || it.is_unstable(db)
                            || ret_ty.is_reference()
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
                                .take(MAX_VARIATIONS)
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
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, goal).then(|| trees))
        .flatten()
}

/// Impl method tactic
///
/// Attempts to to call methods on types from lookup table.
/// This includes both functions from direct impl blocks as well as functions from traits.
///
/// # Arguments
/// * `db` - HIR database
/// * `module` - Module where the term search target location
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `goal` - Term search target type
pub(super) fn impl_method<'a>(
    db: &'a dyn HirDatabase,
    module: &'a Module,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
    goal: &'a Type,
) -> impl Iterator<Item = TypeTree> + 'a {
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
        .filter_map(|(imp, ty, it)| {
            let fn_generics = db.generic_params(it.id.into());
            let imp_generics = db.generic_params(imp.id.into());

            // Ignore impl if it has const type arguments
            if fn_generics
                .type_or_consts
                .values()
                .any(|it| matches!(it, TypeOrConstParamData::ConstParamData(_)))
                || imp_generics
                    .type_or_consts
                    .values()
                    .any(|it| matches!(it, TypeOrConstParamData::ConstParamData(_)))
            {
                return None;
            }

            // Ignore all functions that have something to do with lifetimes as we don't check them
            if !fn_generics.lifetimes.is_empty() {
                return None;
            }

            // Ignore functions without self param
            if !it.has_self_param(db) {
                return None;
            }

            // Filter out private and unsafe functions
            if !it.is_visible_from(db, *module) || it.is_unsafe_to_call(db) || it.is_unstable(db) {
                return None;
            }

            // Ignore bigger number of generics for now as they kill the performance
            if imp_generics.type_or_consts.len() + fn_generics.type_or_consts.len() > 0 {
                return None;
            }

            let generic_params = lookup
                .iter_types()
                .collect::<Vec<_>>() // Force take ownership
                .into_iter()
                .permutations(imp_generics.type_or_consts.len() + fn_generics.type_or_consts.len());

            let trees: Vec<_> = generic_params
                .filter_map(|generics| {
                    let ret_ty = it.ret_type_with_generics(
                        db,
                        ty.type_arguments().chain(generics.iter().cloned()),
                    );
                    // Filter out functions that return references
                    if ret_ty.is_reference() || ret_ty.is_raw_ptr() {
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
                        .take(MAX_VARIATIONS)
                        .map(|params| TypeTree::Function { func: it, generics: Vec::new(), params })
                        .collect();

                    lookup.insert(ret_ty.clone(), fn_trees.iter().cloned());
                    Some((ret_ty, fn_trees))
                })
                .collect();
            Some(trees)
        })
        .flatten()
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, goal).then(|| trees))
        .flatten()
}

/// Struct projection tactic
///
/// Attempts different struct fields
///
/// # Arguments
/// * `db` - HIR database
/// * `module` - Module where the term search target location
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `goal` - Term search target type
pub(super) fn struct_projection<'a>(
    db: &'a dyn HirDatabase,
    module: &'a Module,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
    goal: &'a Type,
) -> impl Iterator<Item = TypeTree> + 'a {
    lookup
        .new_types(NewTypesKey::StructProjection)
        .into_iter()
        .map(|ty| (ty.clone(), lookup.find(db, &ty).expect("TypeTree not in lookup")))
        .flat_map(move |(ty, targets)| {
            let module = module.clone();
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
        .filter_map(|(ty, trees)| ty.could_unify_with_deeply(db, goal).then(|| trees))
        .flatten()
}

/// Famous types tactic
///
/// Attempts different values of well known types such as `true` or `false`
///
/// # Arguments
/// * `db` - HIR database
/// * `module` - Module where the term search target location
/// * `defs` - Set of items in scope at term search target location
/// * `lookup` - Lookup table for types
/// * `goal` - Term search target type
pub(super) fn famous_types<'a>(
    db: &'a dyn HirDatabase,
    module: &'a Module,
    _defs: &'a FxHashSet<ScopeDef>,
    lookup: &'a mut LookupTable,
    goal: &'a Type,
) -> impl Iterator<Item = TypeTree> + 'a {
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
    .filter(|tt| tt.ty(db).could_unify_with_deeply(db, goal))
}
