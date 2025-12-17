//! Handling of opaque types, detection of defining scope and hidden type.

use hir_def::{
    AssocItemId, AssocItemLoc, DefWithBodyId, FunctionId, HasModule, ItemContainerId, TypeAliasId,
};
use hir_expand::name::Name;
use la_arena::ArenaMap;
use rustc_type_ir::inherent::Ty as _;
use syntax::ast;

use crate::{
    ImplTraitId, InferenceResult,
    db::{HirDatabase, InternedOpaqueTyId},
    lower::{ImplTraitIdx, ImplTraits},
    next_solver::{
        DbInterner, EarlyBinder, ErrorGuaranteed, SolverDefId, Ty, TypingMode,
        infer::{DbInternerInferExt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

pub(crate) fn opaque_types_defined_by(
    db: &dyn HirDatabase,
    def_id: DefWithBodyId,
    result: &mut Vec<SolverDefId>,
) {
    if let DefWithBodyId::FunctionId(func) = def_id {
        // A function may define its own RPITs.
        extend_with_opaques(
            db,
            ImplTraits::return_type_impl_traits(db, func),
            |opaque_idx| ImplTraitId::ReturnTypeImplTrait(func, opaque_idx),
            result,
        );
    }

    let extend_with_taits = |type_alias| {
        extend_with_opaques(
            db,
            ImplTraits::type_alias_impl_traits(db, type_alias),
            |opaque_idx| ImplTraitId::TypeAliasImplTrait(type_alias, opaque_idx),
            result,
        );
    };

    // Collect opaques from assoc items.
    let extend_with_atpit_from_assoc_items = |assoc_items: &[(Name, AssocItemId)]| {
        assoc_items
            .iter()
            .filter_map(|&(_, assoc_id)| match assoc_id {
                AssocItemId::TypeAliasId(it) => Some(it),
                AssocItemId::FunctionId(_) | AssocItemId::ConstId(_) => None,
            })
            .for_each(extend_with_taits);
    };
    let extend_with_atpit_from_container = |container| match container {
        ItemContainerId::ImplId(impl_id) => {
            if db.impl_signature(impl_id).target_trait.is_some() {
                extend_with_atpit_from_assoc_items(&impl_id.impl_items(db).items);
            }
        }
        ItemContainerId::TraitId(trait_id) => {
            extend_with_atpit_from_assoc_items(&trait_id.trait_items(db).items);
        }
        _ => {}
    };
    match def_id {
        DefWithBodyId::ConstId(id) => extend_with_atpit_from_container(id.loc(db).container),
        DefWithBodyId::FunctionId(id) => extend_with_atpit_from_container(id.loc(db).container),
        DefWithBodyId::StaticId(_) | DefWithBodyId::VariantId(_) => {}
    }

    // FIXME: Collect opaques from `#[define_opaque]`.

    fn extend_with_opaques<'db>(
        db: &'db dyn HirDatabase,
        opaques: &Option<Box<EarlyBinder<'db, ImplTraits<'db>>>>,
        mut make_impl_trait: impl FnMut(ImplTraitIdx<'db>) -> ImplTraitId<'db>,
        result: &mut Vec<SolverDefId>,
    ) {
        if let Some(opaques) = opaques {
            for (opaque_idx, _) in (**opaques).as_ref().skip_binder().impl_traits.iter() {
                let opaque_id = InternedOpaqueTyId::new(db, make_impl_trait(opaque_idx));
                result.push(opaque_id.into());
            }
        }
    }
}

// These are firewall queries to prevent drawing dependencies between infers:

#[salsa::tracked(returns(ref), unsafe(non_update_return_type))]
pub(crate) fn rpit_hidden_types<'db>(
    db: &'db dyn HirDatabase,
    function: FunctionId,
) -> ArenaMap<ImplTraitIdx<'db>, EarlyBinder<'db, Ty<'db>>> {
    let infer = InferenceResult::for_body(db, function.into());
    let mut result = ArenaMap::new();
    for (opaque, hidden_type) in infer.return_position_impl_trait_types(db) {
        result.insert(opaque, EarlyBinder::bind(hidden_type));
    }
    result.shrink_to_fit();
    result
}

#[salsa::tracked(returns(ref), unsafe(non_update_return_type))]
pub(crate) fn tait_hidden_types<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> ArenaMap<ImplTraitIdx<'db>, EarlyBinder<'db, Ty<'db>>> {
    // Call this first, to not perform redundant work if there are no TAITs.
    let Some(taits_count) = ImplTraits::type_alias_impl_traits(db, type_alias)
        .as_deref()
        .map(|taits| taits.as_ref().skip_binder().impl_traits.len())
    else {
        return ArenaMap::new();
    };

    let loc = type_alias.loc(db);
    let module = loc.module(db);
    let interner = DbInterner::new_with(db, module.krate(db));
    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());
    let mut ocx = ObligationCtxt::new(&infcx);
    let cause = ObligationCause::dummy();
    let param_env = db.trait_environment(type_alias.into());

    let defining_bodies = tait_defining_bodies(db, &loc);

    let mut result = ArenaMap::with_capacity(taits_count);
    for defining_body in defining_bodies {
        let infer = InferenceResult::for_body(db, defining_body);
        for (&opaque, &hidden_type) in &infer.type_of_opaque {
            let ImplTraitId::TypeAliasImplTrait(opaque_owner, opaque_idx) = opaque.loc(db) else {
                continue;
            };
            if opaque_owner != type_alias {
                continue;
            }
            // In the presence of errors, we attempt to create a unified type from all
            // types. rustc doesn't do that, but this should improve the experience.
            let hidden_type = infcx.insert_type_vars(hidden_type);
            match result.entry(opaque_idx) {
                la_arena::Entry::Vacant(entry) => {
                    entry.insert(EarlyBinder::bind(hidden_type));
                }
                la_arena::Entry::Occupied(entry) => {
                    _ = ocx.eq(&cause, param_env, entry.get().instantiate_identity(), hidden_type);
                }
            }
        }
    }

    _ = ocx.try_evaluate_obligations();

    // Fill missing entries.
    for idx in 0..taits_count {
        let idx = la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(idx as u32));
        match result.entry(idx) {
            la_arena::Entry::Vacant(entry) => {
                entry.insert(EarlyBinder::bind(Ty::new_error(interner, ErrorGuaranteed)));
            }
            la_arena::Entry::Occupied(mut entry) => {
                *entry.get_mut() = entry.get().map_bound(|hidden_type| {
                    infcx.resolve_vars_if_possible(hidden_type).replace_infer_with_error(interner)
                });
            }
        }
    }

    result
}

fn tait_defining_bodies(
    db: &dyn HirDatabase,
    loc: &AssocItemLoc<ast::TypeAlias>,
) -> Vec<DefWithBodyId> {
    let from_assoc_items = |assoc_items: &[(Name, AssocItemId)]| {
        // Associated Type Position Impl Trait.
        assoc_items
            .iter()
            .filter_map(|&(_, assoc_id)| match assoc_id {
                AssocItemId::FunctionId(it) => Some(it.into()),
                AssocItemId::ConstId(it) => Some(it.into()),
                AssocItemId::TypeAliasId(_) => None,
            })
            .collect()
    };
    match loc.container {
        ItemContainerId::ImplId(impl_id) => {
            if db.impl_signature(impl_id).target_trait.is_some() {
                return from_assoc_items(&impl_id.impl_items(db).items);
            }
        }
        ItemContainerId::TraitId(trait_id) => {
            return from_assoc_items(&trait_id.trait_items(db).items);
        }
        _ => {}
    }

    // FIXME: Support general TAITs, or decisively decide not to.
    Vec::new()
}
