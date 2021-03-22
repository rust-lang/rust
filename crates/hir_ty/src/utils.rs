//! Helper functions for working with def, which don't need to be a separate
//! query, but can't be computed directly from `*Data` (ie, which need a `db`).
use std::sync::Arc;

use hir_def::{
    adt::VariantData,
    db::DefDatabase,
    generics::{
        GenericParams, TypeParamData, TypeParamProvenance, WherePredicate, WherePredicateTypeTarget,
    },
    path::Path,
    resolver::{HasResolver, TypeNs},
    type_ref::TypeRef,
    AssocContainerId, GenericDefId, Lookup, TraitId, TypeAliasId, TypeParamId, VariantId,
};
use hir_expand::name::{name, Name};

use crate::{db::HirDatabase, TraitRef, WhereClause};

fn direct_super_traits(db: &dyn DefDatabase, trait_: TraitId) -> Vec<TraitId> {
    let resolver = trait_.resolver(db);
    // returning the iterator directly doesn't easily work because of
    // lifetime problems, but since there usually shouldn't be more than a
    // few direct traits this should be fine (we could even use some kind of
    // SmallVec if performance is a concern)
    let generic_params = db.generic_params(trait_.into());
    let trait_self = generic_params.find_trait_self_param();
    generic_params
        .where_predicates
        .iter()
        .filter_map(|pred| match pred {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => match target {
                WherePredicateTypeTarget::TypeRef(TypeRef::Path(p))
                    if p == &Path::from(name![Self]) =>
                {
                    bound.as_path()
                }
                WherePredicateTypeTarget::TypeParam(local_id) if Some(*local_id) == trait_self => {
                    bound.as_path()
                }
                _ => None,
            },
            WherePredicate::Lifetime { .. } => None,
        })
        .filter_map(|path| match resolver.resolve_path_in_type_ns_fully(db, path.mod_path()) {
            Some(TypeNs::TraitId(t)) => Some(t),
            _ => None,
        })
        .collect()
}

fn direct_super_trait_refs(db: &dyn HirDatabase, trait_ref: &TraitRef) -> Vec<TraitRef> {
    // returning the iterator directly doesn't easily work because of
    // lifetime problems, but since there usually shouldn't be more than a
    // few direct traits this should be fine (we could even use some kind of
    // SmallVec if performance is a concern)
    let generic_params = db.generic_params(trait_ref.hir_trait_id().into());
    let trait_self = match generic_params.find_trait_self_param() {
        Some(p) => TypeParamId { parent: trait_ref.hir_trait_id().into(), local_id: p },
        None => return Vec::new(),
    };
    db.generic_predicates_for_param(trait_self)
        .iter()
        .filter_map(|pred| {
            pred.as_ref().filter_map(|pred| match pred.skip_binders() {
                WhereClause::Implemented(tr) => Some(tr.clone()),
                _ => None,
            })
        })
        .map(|pred| pred.subst(&trait_ref.substitution))
        .collect()
}

/// Returns an iterator over the whole super trait hierarchy (including the
/// trait itself).
pub(super) fn all_super_traits(db: &dyn DefDatabase, trait_: TraitId) -> Vec<TraitId> {
    // we need to take care a bit here to avoid infinite loops in case of cycles
    // (i.e. if we have `trait A: B; trait B: A;`)
    let mut result = vec![trait_];
    let mut i = 0;
    while i < result.len() {
        let t = result[i];
        // yeah this is quadratic, but trait hierarchies should be flat
        // enough that this doesn't matter
        for tt in direct_super_traits(db, t) {
            if !result.contains(&tt) {
                result.push(tt);
            }
        }
        i += 1;
    }
    result
}

/// Given a trait ref (`Self: Trait`), builds all the implied trait refs for
/// super traits. The original trait ref will be included. So the difference to
/// `all_super_traits` is that we keep track of type parameters; for example if
/// we have `Self: Trait<u32, i32>` and `Trait<T, U>: OtherTrait<U>` we'll get
/// `Self: OtherTrait<i32>`.
pub(super) fn all_super_trait_refs(db: &dyn HirDatabase, trait_ref: TraitRef) -> Vec<TraitRef> {
    // we need to take care a bit here to avoid infinite loops in case of cycles
    // (i.e. if we have `trait A: B; trait B: A;`)
    let mut result = vec![trait_ref];
    let mut i = 0;
    while i < result.len() {
        let t = &result[i];
        // yeah this is quadratic, but trait hierarchies should be flat
        // enough that this doesn't matter
        for tt in direct_super_trait_refs(db, t) {
            if !result.iter().any(|tr| tr.trait_id == tt.trait_id) {
                result.push(tt);
            }
        }
        i += 1;
    }
    result
}

pub(super) fn associated_type_by_name_including_super_traits(
    db: &dyn HirDatabase,
    trait_ref: TraitRef,
    name: &Name,
) -> Option<(TraitRef, TypeAliasId)> {
    all_super_trait_refs(db, trait_ref).into_iter().find_map(|t| {
        let assoc_type = db.trait_data(t.hir_trait_id()).associated_type_by_name(name)?;
        Some((t, assoc_type))
    })
}

pub(super) fn variant_data(db: &dyn DefDatabase, var: VariantId) -> Arc<VariantData> {
    match var {
        VariantId::StructId(it) => db.struct_data(it).variant_data.clone(),
        VariantId::UnionId(it) => db.union_data(it).variant_data.clone(),
        VariantId::EnumVariantId(it) => {
            db.enum_data(it.parent).variants[it.local_id].variant_data.clone()
        }
    }
}

/// Helper for mutating `Arc<[T]>` (i.e. `Arc::make_mut` for Arc slices).
/// The underlying values are cloned if there are other strong references.
pub(crate) fn make_mut_slice<T: Clone>(a: &mut Arc<[T]>) -> &mut [T] {
    if Arc::get_mut(a).is_none() {
        *a = a.iter().cloned().collect();
    }
    Arc::get_mut(a).unwrap()
}

pub(crate) fn generics(db: &dyn DefDatabase, def: GenericDefId) -> Generics {
    let parent_generics = parent_generic_def(db, def).map(|def| Box::new(generics(db, def)));
    Generics { def, params: db.generic_params(def), parent_generics }
}

#[derive(Debug)]
pub(crate) struct Generics {
    def: GenericDefId,
    pub(crate) params: Arc<GenericParams>,
    parent_generics: Option<Box<Generics>>,
}

impl Generics {
    pub(crate) fn iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (TypeParamId, &'a TypeParamData)> + 'a {
        self.parent_generics
            .as_ref()
            .into_iter()
            .flat_map(|it| {
                it.params
                    .types
                    .iter()
                    .map(move |(local_id, p)| (TypeParamId { parent: it.def, local_id }, p))
            })
            .chain(
                self.params
                    .types
                    .iter()
                    .map(move |(local_id, p)| (TypeParamId { parent: self.def, local_id }, p)),
            )
    }

    pub(crate) fn iter_parent<'a>(
        &'a self,
    ) -> impl Iterator<Item = (TypeParamId, &'a TypeParamData)> + 'a {
        self.parent_generics.as_ref().into_iter().flat_map(|it| {
            it.params
                .types
                .iter()
                .map(move |(local_id, p)| (TypeParamId { parent: it.def, local_id }, p))
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len_split().0
    }

    /// (total, parents, child)
    pub(crate) fn len_split(&self) -> (usize, usize, usize) {
        let parent = self.parent_generics.as_ref().map_or(0, |p| p.len());
        let child = self.params.types.len();
        (parent + child, parent, child)
    }

    /// (parent total, self param, type param list, impl trait)
    pub(crate) fn provenance_split(&self) -> (usize, usize, usize, usize) {
        let parent = self.parent_generics.as_ref().map_or(0, |p| p.len());
        let self_params = self
            .params
            .types
            .iter()
            .filter(|(_, p)| p.provenance == TypeParamProvenance::TraitSelf)
            .count();
        let list_params = self
            .params
            .types
            .iter()
            .filter(|(_, p)| p.provenance == TypeParamProvenance::TypeParamList)
            .count();
        let impl_trait_params = self
            .params
            .types
            .iter()
            .filter(|(_, p)| p.provenance == TypeParamProvenance::ArgumentImplTrait)
            .count();
        (parent, self_params, list_params, impl_trait_params)
    }

    pub(crate) fn param_idx(&self, param: TypeParamId) -> Option<usize> {
        Some(self.find_param(param)?.0)
    }

    fn find_param(&self, param: TypeParamId) -> Option<(usize, &TypeParamData)> {
        if param.parent == self.def {
            let (idx, (_local_id, data)) = self
                .params
                .types
                .iter()
                .enumerate()
                .find(|(_, (idx, _))| *idx == param.local_id)
                .unwrap();
            let (_total, parent_len, _child) = self.len_split();
            Some((parent_len + idx, data))
        } else {
            self.parent_generics.as_ref().and_then(|g| g.find_param(param))
        }
    }
}

fn parent_generic_def(db: &dyn DefDatabase, def: GenericDefId) -> Option<GenericDefId> {
    let container = match def {
        GenericDefId::FunctionId(it) => it.lookup(db).container,
        GenericDefId::TypeAliasId(it) => it.lookup(db).container,
        GenericDefId::ConstId(it) => it.lookup(db).container,
        GenericDefId::EnumVariantId(it) => return Some(it.parent.into()),
        GenericDefId::AdtId(_) | GenericDefId::TraitId(_) | GenericDefId::ImplId(_) => return None,
    };

    match container {
        AssocContainerId::ImplId(it) => Some(it.into()),
        AssocContainerId::TraitId(it) => Some(it.into()),
        AssocContainerId::ModuleId(_) => None,
    }
}
