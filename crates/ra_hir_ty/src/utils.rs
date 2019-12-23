//! Helper functions for working with def, which don't need to be a separate
//! query, but can't be computed directly from `*Data` (ie, which need a `db`).
use std::sync::Arc;

use hir_def::{
    adt::VariantData,
    db::DefDatabase,
    generics::{GenericParams, TypeParamData},
    path::Path,
    resolver::{HasResolver, TypeNs},
    type_ref::TypeRef,
    AssocContainerId, GenericDefId, Lookup, TraitId, TypeAliasId, TypeParamId, VariantId,
};
use hir_expand::name::{name, Name};

fn direct_super_traits(db: &impl DefDatabase, trait_: TraitId) -> Vec<TraitId> {
    let resolver = trait_.resolver(db);
    // returning the iterator directly doesn't easily work because of
    // lifetime problems, but since there usually shouldn't be more than a
    // few direct traits this should be fine (we could even use some kind of
    // SmallVec if performance is a concern)
    db.generic_params(trait_.into())
        .where_predicates
        .iter()
        .filter_map(|pred| match &pred.type_ref {
            TypeRef::Path(p) if p == &Path::from(name![Self]) => pred.bound.as_path(),
            _ => None,
        })
        .filter_map(|path| match resolver.resolve_path_in_type_ns_fully(db, path.mod_path()) {
            Some(TypeNs::TraitId(t)) => Some(t),
            _ => None,
        })
        .collect()
}

/// Returns an iterator over the whole super trait hierarchy (including the
/// trait itself).
pub(super) fn all_super_traits(db: &impl DefDatabase, trait_: TraitId) -> Vec<TraitId> {
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

pub(super) fn associated_type_by_name_including_super_traits(
    db: &impl DefDatabase,
    trait_: TraitId,
    name: &Name,
) -> Option<TypeAliasId> {
    all_super_traits(db, trait_)
        .into_iter()
        .find_map(|t| db.trait_data(t).associated_type_by_name(name))
}

pub(super) fn variant_data(db: &impl DefDatabase, var: VariantId) -> Arc<VariantData> {
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

pub(crate) fn generics(db: &impl DefDatabase, def: GenericDefId) -> Generics {
    let parent_generics = parent_generic_def(db, def).map(|def| Box::new(generics(db, def)));
    Generics { def, params: db.generic_params(def), parent_generics }
}

pub(crate) struct Generics {
    def: GenericDefId,
    pub(crate) params: Arc<GenericParams>,
    parent_generics: Option<Box<Generics>>,
}

impl Generics {
    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = (u32, &'a TypeParamData)> + 'a {
        self.parent_generics
            .as_ref()
            .into_iter()
            .flat_map(|it| it.params.types.iter())
            .chain(self.params.types.iter())
            .enumerate()
            .map(|(i, (_local_id, p))| (i as u32, p))
    }

    pub(crate) fn iter_parent<'a>(&'a self) -> impl Iterator<Item = (u32, &'a TypeParamData)> + 'a {
        self.parent_generics
            .as_ref()
            .into_iter()
            .flat_map(|it| it.params.types.iter())
            .enumerate()
            .map(|(i, (_local_id, p))| (i as u32, p))
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
    pub(crate) fn param_idx(&self, param: TypeParamId) -> u32 {
        self.find_param(param).0
    }
    pub(crate) fn param_name(&self, param: TypeParamId) -> Name {
        self.find_param(param).1.name.clone()
    }
    fn find_param(&self, param: TypeParamId) -> (u32, &TypeParamData) {
        if param.parent == self.def {
            let (idx, (_local_id, data)) = self
                .params
                .types
                .iter()
                .enumerate()
                .find(|(_, (idx, _))| *idx == param.local_id)
                .unwrap();
            let (_total, parent_len, _child) = self.len_split();
            return ((parent_len + idx) as u32, data);
        }
        self.parent_generics.as_ref().unwrap().find_param(param)
    }
}

fn parent_generic_def(db: &impl DefDatabase, def: GenericDefId) -> Option<GenericDefId> {
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
        AssocContainerId::ContainerId(_) => None,
    }
}
