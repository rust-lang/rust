//! Helper functions for working with def, which don't need to be a separate
//! query, but can't be computed directly from `*Data` (ie, which need a `db`).

use std::{hash::Hash, iter};

use base_db::CrateId;
use chalk_ir::{
    fold::{FallibleTypeFolder, Shift},
    DebruijnIndex,
};
use hir_def::{
    db::DefDatabase,
    generics::{WherePredicate, WherePredicateTypeTarget},
    lang_item::LangItem,
    resolver::{HasResolver, TypeNs},
    type_ref::{TraitBoundModifier, TypeRef},
    EnumId, EnumVariantId, FunctionId, Lookup, OpaqueInternableThing, TraitId, TypeAliasId,
    TypeOrConstParamId,
};
use hir_expand::name::Name;
use rustc_abi::TargetDataLayout;
use rustc_hash::FxHashSet;
use smallvec::{smallvec, SmallVec};
use span::Edition;
use stdx::never;

use crate::{
    consteval::unknown_const,
    db::HirDatabase,
    layout::{Layout, TagEncoding},
    mir::pad16,
    ChalkTraitId, Const, ConstScalar, GenericArg, Interner, Substitution, TargetFeatures, TraitRef,
    TraitRefExt, Ty, WhereClause,
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

/// Returns an iterator over the direct super traits (including the trait itself).
pub fn direct_super_traits(db: &dyn DefDatabase, trait_: TraitId) -> SmallVec<[TraitId; 4]> {
    let mut result = smallvec![trait_];
    direct_super_traits_cb(db, trait_, |tt| {
        if !result.contains(&tt) {
            result.push(tt);
        }
    });
    result
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
        direct_super_traits_cb(db, t, |tt| {
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
    SuperTraits { db, seen, stack: vec![trait_ref] }.find_map(cb)
}

struct SuperTraits<'a> {
    db: &'a dyn HirDatabase,
    stack: Vec<TraitRef>,
    seen: FxHashSet<ChalkTraitId>,
}

impl SuperTraits<'_> {
    fn elaborate(&mut self, trait_ref: &TraitRef) {
        direct_super_trait_refs(self.db, trait_ref, |trait_ref| {
            if !self.seen.contains(&trait_ref.trait_id) {
                self.stack.push(trait_ref);
            }
        });
    }
}

impl Iterator for SuperTraits<'_> {
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

pub(super) fn elaborate_clause_supertraits(
    db: &dyn HirDatabase,
    clauses: impl Iterator<Item = WhereClause>,
) -> ClauseElaborator<'_> {
    let mut elaborator = ClauseElaborator { db, stack: Vec::new(), seen: FxHashSet::default() };
    elaborator.extend_deduped(clauses);

    elaborator
}

pub(super) struct ClauseElaborator<'a> {
    db: &'a dyn HirDatabase,
    stack: Vec<WhereClause>,
    seen: FxHashSet<WhereClause>,
}

impl ClauseElaborator<'_> {
    fn extend_deduped(&mut self, clauses: impl IntoIterator<Item = WhereClause>) {
        self.stack.extend(clauses.into_iter().filter(|c| self.seen.insert(c.clone())))
    }

    fn elaborate_supertrait(&mut self, clause: &WhereClause) {
        if let WhereClause::Implemented(trait_ref) = clause {
            direct_super_trait_refs(self.db, trait_ref, |t| {
                let clause = WhereClause::Implemented(t);
                if self.seen.insert(clause.clone()) {
                    self.stack.push(clause);
                }
            });
        }
    }
}

impl Iterator for ClauseElaborator<'_> {
    type Item = WhereClause;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.stack.pop() {
            self.elaborate_supertrait(&next);
            Some(next)
        } else {
            None
        }
    }
}

fn direct_super_traits_cb(db: &dyn DefDatabase, trait_: TraitId, cb: impl FnMut(TraitId)) {
    let resolver = trait_.resolver(db);
    let generic_params = db.generic_params(trait_.into());
    let trait_self = generic_params.trait_self_param();
    generic_params
        .where_predicates()
        .filter_map(|pred| match pred {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let is_trait = match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => {
                        match &generic_params.types_map[*type_ref] {
                            TypeRef::Path(p) => p.is_self_type(),
                            _ => false,
                        }
                    }
                    WherePredicateTypeTarget::TypeOrConstParam(local_id) => {
                        Some(*local_id) == trait_self
                    }
                };
                match is_trait {
                    true => bound.as_path(&generic_params.types_map),
                    false => None,
                }
            }
            WherePredicate::Lifetime { .. } => None,
        })
        .filter(|(_, bound_modifier)| matches!(bound_modifier, TraitBoundModifier::None))
        .filter_map(|(path, _)| match resolver.resolve_path_in_type_ns_fully(db, path) {
            Some(TypeNs::TraitId(t)) => Some(t),
            _ => None,
        })
        .for_each(cb);
}

fn direct_super_trait_refs(db: &dyn HirDatabase, trait_ref: &TraitRef, cb: impl FnMut(TraitRef)) {
    let generic_params = db.generic_params(trait_ref.hir_trait_id().into());
    let trait_self = match generic_params.trait_self_param() {
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

/// It is a bit different from the rustc equivalent. Currently it stores:
/// - 0: the function signature, encoded as a function pointer type
/// - 1..n: generics of the parent
///
/// and it doesn't store the closure types and fields.
///
/// Codes should not assume this ordering, and should always use methods available
/// on this struct for retrieving, and `TyBuilder::substs_for_closure` for creating.
pub(crate) struct ClosureSubst<'a>(pub(crate) &'a Substitution);

impl<'a> ClosureSubst<'a> {
    pub(crate) fn parent_subst(&self) -> &'a [GenericArg] {
        match self.0.as_slice(Interner) {
            [_, x @ ..] => x,
            _ => {
                never!("Closure missing parameter");
                &[]
            }
        }
    }

    pub(crate) fn sig_ty(&self) -> &'a Ty {
        match self.0.as_slice(Interner) {
            [x, ..] => x.assert_ty_ref(Interner),
            _ => {
                unreachable!("Closure missing sig_ty parameter");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsafety {
    Safe,
    Unsafe,
    /// A lint.
    DeprecatedSafe2024,
}

pub fn is_fn_unsafe_to_call(
    db: &dyn HirDatabase,
    func: FunctionId,
    caller_target_features: &TargetFeatures,
    call_edition: Edition,
) -> Unsafety {
    let data = db.function_data(func);
    if data.is_unsafe() {
        return Unsafety::Unsafe;
    }

    if data.has_target_feature() {
        // RFC 2396 <https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html>.
        let callee_target_features =
            TargetFeatures::from_attrs_no_implications(&db.attrs(func.into()));
        if !caller_target_features.enabled.is_superset(&callee_target_features.enabled) {
            return Unsafety::Unsafe;
        }
    }

    if data.is_deprecated_safe_2024() {
        if call_edition.at_least_2024() {
            return Unsafety::Unsafe;
        } else {
            return Unsafety::DeprecatedSafe2024;
        }
    }

    let loc = func.lookup(db.upcast());
    match loc.container {
        hir_def::ItemContainerId::ExternBlockId(_block) => {
            // Function in an `extern` block are always unsafe to call, except when
            // it is marked as `safe`.
            if data.is_safe() {
                Unsafety::Safe
            } else {
                Unsafety::Unsafe
            }
        }
        _ => Unsafety::Safe,
    }
}

pub(crate) struct UnevaluatedConstEvaluatorFolder<'a> {
    pub(crate) db: &'a dyn HirDatabase,
}

impl FallibleTypeFolder<Interner> for UnevaluatedConstEvaluatorFolder<'_> {
    type Error = ();

    fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = ()> {
        self
    }

    fn interner(&self) -> Interner {
        Interner
    }

    fn try_fold_const(
        &mut self,
        constant: Const,
        _outer_binder: DebruijnIndex,
    ) -> Result<Const, Self::Error> {
        if let chalk_ir::ConstValue::Concrete(c) = &constant.data(Interner).value {
            if let ConstScalar::UnevaluatedConst(id, subst) = &c.interned {
                if let Ok(eval) = self.db.const_eval(*id, subst.clone(), None) {
                    return Ok(eval);
                } else {
                    return Ok(unknown_const(constant.data(Interner).ty.clone()));
                }
            }
        }
        Ok(constant)
    }
}

pub(crate) fn detect_variant_from_bytes<'a>(
    layout: &'a Layout,
    db: &dyn HirDatabase,
    target_data_layout: &TargetDataLayout,
    b: &[u8],
    e: EnumId,
) -> Option<(EnumVariantId, &'a Layout)> {
    let (var_id, var_layout) = match &layout.variants {
        hir_def::layout::Variants::Empty => unreachable!(),
        hir_def::layout::Variants::Single { index } => {
            (db.enum_data(e).variants[index.0].0, layout)
        }
        hir_def::layout::Variants::Multiple { tag, tag_encoding, variants, .. } => {
            let size = tag.size(target_data_layout).bytes_usize();
            let offset = layout.fields.offset(0).bytes_usize(); // The only field on enum variants is the tag field
            let tag = i128::from_le_bytes(pad16(&b[offset..offset + size], false));
            match tag_encoding {
                TagEncoding::Direct => {
                    let (var_idx, layout) =
                        variants.iter_enumerated().find_map(|(var_idx, v)| {
                            let def = db.enum_data(e).variants[var_idx.0].0;
                            (db.const_eval_discriminant(def) == Ok(tag)).then_some((def, v))
                        })?;
                    (var_idx, layout)
                }
                TagEncoding::Niche { untagged_variant, niche_start, .. } => {
                    let candidate_tag = tag.wrapping_sub(*niche_start as i128) as usize;
                    let variant = variants
                        .iter_enumerated()
                        .map(|(x, _)| x)
                        .filter(|x| x != untagged_variant)
                        .nth(candidate_tag)
                        .unwrap_or(*untagged_variant);
                    (db.enum_data(e).variants[variant.0].0, &variants[variant])
                }
            }
        }
    };
    Some((var_id, var_layout))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct InTypeConstIdMetadata(pub(crate) Ty);

impl OpaqueInternableThing for InTypeConstIdMetadata {
    fn dyn_hash(&self, mut state: &mut dyn std::hash::Hasher) {
        self.hash(&mut state);
    }

    fn dyn_eq(&self, other: &dyn OpaqueInternableThing) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn dyn_clone(&self) -> Box<dyn OpaqueInternableThing> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn box_any(&self) -> Box<dyn std::any::Any> {
        Box::new(self.clone())
    }
}
