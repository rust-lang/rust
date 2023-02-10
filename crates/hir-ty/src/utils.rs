//! Helper functions for working with def, which don't need to be a separate
//! query, but can't be computed directly from `*Data` (ie, which need a `db`).

use std::iter;

use base_db::CrateId;
use chalk_ir::{cast::Cast, fold::Shift, BoundVar, DebruijnIndex};
use hir_def::{
    db::DefDatabase,
    generics::{
        GenericParams, TypeOrConstParamData, TypeParamProvenance, WherePredicate,
        WherePredicateTypeTarget,
    },
    lang_item::LangItem,
    resolver::{HasResolver, TypeNs},
    type_ref::{TraitBoundModifier, TypeRef},
    ConstParamId, FunctionId, GenericDefId, ItemContainerId, Lookup, TraitId, TypeAliasId,
    TypeOrConstParamId, TypeParamId,
};
use hir_expand::name::Name;
use intern::Interned;
use itertools::Either;
use rustc_hash::FxHashSet;
use smallvec::{smallvec, SmallVec};

use crate::{
    db::HirDatabase, ChalkTraitId, Interner, Substitution, TraitRef, TraitRefExt, WhereClause,
};

pub(crate) fn fn_traits(
    db: &dyn DefDatabase,
    krate: CrateId,
) -> impl Iterator<Item = TraitId> + '_ {
    [LangItem::Fn, LangItem::FnMut, LangItem::FnOnce]
        .into_iter()
        .filter_map(move |lang| db.lang_item(krate, lang))
        .flat_map(|it| it.as_trait())
}

/// Returns an iterator over the whole super trait hierarchy (including the
/// trait itself).
pub fn all_super_traits(db: &dyn DefDatabase, trait_: TraitId) -> SmallVec<[TraitId; 4]> {
    // we need to take care a bit here to avoid infinite loops in case of cycles
    // (i.e. if we have `trait A: B; trait B: A;`)

    let mut result = smallvec![trait_];
    let mut i = 0;
    while let Some(&t) = result.get(i) {
        // yeah this is quadratic, but trait hierarchies should be flat
        // enough that this doesn't matter
        direct_super_traits(db, t, |tt| {
            if !result.contains(&tt) {
                result.push(tt);
            }
        });
        i += 1;
    }
    result
}

/// Given a trait ref (`Self: Trait`), builds all the implied trait refs for
/// super traits. The original trait ref will be included. So the difference to
/// `all_super_traits` is that we keep track of type parameters; for example if
/// we have `Self: Trait<u32, i32>` and `Trait<T, U>: OtherTrait<U>` we'll get
/// `Self: OtherTrait<i32>`.
pub(super) fn all_super_trait_refs<T>(
    db: &dyn HirDatabase,
    trait_ref: TraitRef,
    cb: impl FnMut(TraitRef) -> Option<T>,
) -> Option<T> {
    let seen = iter::once(trait_ref.trait_id).collect();
    let mut stack = Vec::new();
    stack.push(trait_ref);
    SuperTraits { db, seen, stack }.find_map(cb)
}

struct SuperTraits<'a> {
    db: &'a dyn HirDatabase,
    stack: Vec<TraitRef>,
    seen: FxHashSet<ChalkTraitId>,
}

impl<'a> SuperTraits<'a> {
    fn elaborate(&mut self, trait_ref: &TraitRef) {
        direct_super_trait_refs(self.db, trait_ref, |trait_ref| {
            if !self.seen.contains(&trait_ref.trait_id) {
                self.stack.push(trait_ref);
            }
        });
    }
}

impl<'a> Iterator for SuperTraits<'a> {
    type Item = TraitRef;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.stack.pop() {
            self.elaborate(&next);
            Some(next)
        } else {
            None
        }
    }
}

fn direct_super_traits(db: &dyn DefDatabase, trait_: TraitId, cb: impl FnMut(TraitId)) {
    let resolver = trait_.resolver(db);
    let generic_params = db.generic_params(trait_.into());
    let trait_self = generic_params.find_trait_self_param();
    generic_params
        .where_predicates
        .iter()
        .filter_map(|pred| match pred {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let is_trait = match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => match &**type_ref {
                        TypeRef::Path(p) => p.is_self_type(),
                        _ => false,
                    },
                    WherePredicateTypeTarget::TypeOrConstParam(local_id) => {
                        Some(*local_id) == trait_self
                    }
                };
                match is_trait {
                    true => bound.as_path(),
                    false => None,
                }
            }
            WherePredicate::Lifetime { .. } => None,
        })
        .filter(|(_, bound_modifier)| matches!(bound_modifier, TraitBoundModifier::None))
        .filter_map(|(path, _)| match resolver.resolve_path_in_type_ns_fully(db, path.mod_path()) {
            Some(TypeNs::TraitId(t)) => Some(t),
            _ => None,
        })
        .for_each(cb);
}

fn direct_super_trait_refs(db: &dyn HirDatabase, trait_ref: &TraitRef, cb: impl FnMut(TraitRef)) {
    let generic_params = db.generic_params(trait_ref.hir_trait_id().into());
    let trait_self = match generic_params.find_trait_self_param() {
        Some(p) => TypeOrConstParamId { parent: trait_ref.hir_trait_id().into(), local_id: p },
        None => return,
    };
    db.generic_predicates_for_param(trait_self.parent, trait_self, None)
        .iter()
        .filter_map(|pred| {
            pred.as_ref().filter_map(|pred| match pred.skip_binders() {
                // FIXME: how to correctly handle higher-ranked bounds here?
                WhereClause::Implemented(tr) => Some(
                    tr.clone()
                        .shifted_out_to(Interner, DebruijnIndex::ONE)
                        .expect("FIXME unexpected higher-ranked trait bound"),
                ),
                _ => None,
            })
        })
        .map(|pred| pred.substitute(Interner, &trait_ref.substitution))
        .for_each(cb);
}

pub(super) fn associated_type_by_name_including_super_traits(
    db: &dyn HirDatabase,
    trait_ref: TraitRef,
    name: &Name,
) -> Option<(TraitRef, TypeAliasId)> {
    all_super_trait_refs(db, trait_ref, |t| {
        let assoc_type = db.trait_data(t.hir_trait_id()).associated_type_by_name(name)?;
        Some((t, assoc_type))
    })
}

pub(crate) fn generics(db: &dyn DefDatabase, def: GenericDefId) -> Generics {
    let parent_generics = parent_generic_def(db, def).map(|def| Box::new(generics(db, def)));
    Generics { def, params: db.generic_params(def), parent_generics }
}

#[derive(Debug)]
pub(crate) struct Generics {
    def: GenericDefId,
    pub(crate) params: Interned<GenericParams>,
    parent_generics: Option<Box<Generics>>,
}

impl Generics {
    pub(crate) fn iter_id(&self) -> impl Iterator<Item = Either<TypeParamId, ConstParamId>> + '_ {
        self.iter().map(|(id, data)| match data {
            TypeOrConstParamData::TypeParamData(_) => Either::Left(TypeParamId::from_unchecked(id)),
            TypeOrConstParamData::ConstParamData(_) => {
                Either::Right(ConstParamId::from_unchecked(id))
            }
        })
    }

    /// Iterator over types and const params of self, then parent.
    pub(crate) fn iter<'a>(
        &'a self,
    ) -> impl DoubleEndedIterator<Item = (TypeOrConstParamId, &'a TypeOrConstParamData)> + 'a {
        let to_toc_id = |it: &'a Generics| {
            move |(local_id, p)| (TypeOrConstParamId { parent: it.def, local_id }, p)
        };
        self.params.iter().map(to_toc_id(self)).chain(self.iter_parent())
    }

    /// Iterate over types and const params without parent params.
    pub(crate) fn iter_self<'a>(
        &'a self,
    ) -> impl DoubleEndedIterator<Item = (TypeOrConstParamId, &'a TypeOrConstParamData)> + 'a {
        let to_toc_id = |it: &'a Generics| {
            move |(local_id, p)| (TypeOrConstParamId { parent: it.def, local_id }, p)
        };
        self.params.iter().map(to_toc_id(self))
    }

    /// Iterator over types and const params of parent.
    pub(crate) fn iter_parent(
        &self,
    ) -> impl DoubleEndedIterator<Item = (TypeOrConstParamId, &TypeOrConstParamData)> {
        self.parent_generics().into_iter().flat_map(|it| {
            let to_toc_id =
                move |(local_id, p)| (TypeOrConstParamId { parent: it.def, local_id }, p);
            it.params.iter().map(to_toc_id)
        })
    }

    /// Returns total number of generic parameters in scope, including those from parent.
    pub(crate) fn len(&self) -> usize {
        let parent = self.parent_generics().map_or(0, Generics::len);
        let child = self.params.type_or_consts.len();
        parent + child
    }

    /// Returns numbers of generic parameters excluding those from parent.
    pub(crate) fn len_self(&self) -> usize {
        self.params.type_or_consts.len()
    }

    /// (parent total, self param, type param list, const param list, impl trait)
    pub(crate) fn provenance_split(&self) -> (usize, usize, usize, usize, usize) {
        let mut self_params = 0;
        let mut type_params = 0;
        let mut impl_trait_params = 0;
        let mut const_params = 0;
        self.params.iter().for_each(|(_, data)| match data {
            TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                TypeParamProvenance::TypeParamList => type_params += 1,
                TypeParamProvenance::TraitSelf => self_params += 1,
                TypeParamProvenance::ArgumentImplTrait => impl_trait_params += 1,
            },
            TypeOrConstParamData::ConstParamData(_) => const_params += 1,
        });

        let parent_len = self.parent_generics().map_or(0, Generics::len);
        (parent_len, self_params, type_params, const_params, impl_trait_params)
    }

    pub(crate) fn param_idx(&self, param: TypeOrConstParamId) -> Option<usize> {
        Some(self.find_param(param)?.0)
    }

    fn find_param(&self, param: TypeOrConstParamId) -> Option<(usize, &TypeOrConstParamData)> {
        if param.parent == self.def {
            let (idx, (_local_id, data)) =
                self.params.iter().enumerate().find(|(_, (idx, _))| *idx == param.local_id)?;
            Some((idx, data))
        } else {
            self.parent_generics()
                .and_then(|g| g.find_param(param))
                // Remember that parent parameters come after parameters for self.
                .map(|(idx, data)| (self.len_self() + idx, data))
        }
    }

    pub(crate) fn parent_generics(&self) -> Option<&Generics> {
        self.parent_generics.as_deref()
    }

    /// Returns a Substitution that replaces each parameter by a bound variable.
    pub(crate) fn bound_vars_subst(
        &self,
        db: &dyn HirDatabase,
        debruijn: DebruijnIndex,
    ) -> Substitution {
        Substitution::from_iter(
            Interner,
            self.iter_id().enumerate().map(|(idx, id)| match id {
                Either::Left(_) => BoundVar::new(debruijn, idx).to_ty(Interner).cast(Interner),
                Either::Right(id) => BoundVar::new(debruijn, idx)
                    .to_const(Interner, db.const_param_ty(id))
                    .cast(Interner),
            }),
        )
    }

    /// Returns a Substitution that replaces each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn placeholder_subst(&self, db: &dyn HirDatabase) -> Substitution {
        Substitution::from_iter(
            Interner,
            self.iter_id().map(|id| match id {
                Either::Left(id) => {
                    crate::to_placeholder_idx(db, id.into()).to_ty(Interner).cast(Interner)
                }
                Either::Right(id) => crate::to_placeholder_idx(db, id.into())
                    .to_const(Interner, db.const_param_ty(id))
                    .cast(Interner),
            }),
        )
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
        ItemContainerId::ImplId(it) => Some(it.into()),
        ItemContainerId::TraitId(it) => Some(it.into()),
        ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => None,
    }
}

pub fn is_fn_unsafe_to_call(db: &dyn HirDatabase, func: FunctionId) -> bool {
    let data = db.function_data(func);
    if data.has_unsafe_kw() {
        return true;
    }

    match func.lookup(db.upcast()).container {
        hir_def::ItemContainerId::ExternBlockId(block) => {
            // Function in an `extern` block are always unsafe to call, except when it has
            // `"rust-intrinsic"` ABI there are a few exceptions.
            let id = block.lookup(db.upcast()).id;

            let is_intrinsic =
                id.item_tree(db.upcast())[id.value].abi.as_deref() == Some("rust-intrinsic");

            if is_intrinsic {
                // Intrinsics are unsafe unless they have the rustc_safe_intrinsic attribute
                !data.attrs.by_key("rustc_safe_intrinsic").exists()
            } else {
                // Extern items are always unsafe
                true
            }
        }
        _ => false,
    }
}
