//! Helper functions for working with def, which don't need to be a separate
//! query, but can't be computed directly from `*Data` (ie, which need a `db`).

use std::cell::LazyCell;

use base_db::{
    Crate,
    target::{self, TargetData},
};
use hir_def::{
    EnumId, EnumVariantId, FunctionId, Lookup, TraitId,
    db::DefDatabase,
    hir::generics::WherePredicate,
    lang_item::LangItem,
    resolver::{HasResolver, TypeNs},
    type_ref::{TraitBoundModifier, TypeRef},
};
use intern::sym;
use rustc_abi::TargetDataLayout;
use smallvec::{SmallVec, smallvec};
use span::Edition;

use crate::{
    TargetFeatures,
    db::HirDatabase,
    layout::{Layout, TagEncoding},
    mir::pad16,
};

pub(crate) fn fn_traits(db: &dyn DefDatabase, krate: Crate) -> impl Iterator<Item = TraitId> + '_ {
    [LangItem::Fn, LangItem::FnMut, LangItem::FnOnce]
        .into_iter()
        .filter_map(move |lang| lang.resolve_trait(db, krate))
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

fn direct_super_traits_cb(db: &dyn DefDatabase, trait_: TraitId, cb: impl FnMut(TraitId)) {
    let resolver = LazyCell::new(|| trait_.resolver(db));
    let (generic_params, store) = db.generic_params_and_store(trait_.into());
    let trait_self = generic_params.trait_self_param();
    generic_params
        .where_predicates()
        .iter()
        .filter_map(|pred| match pred {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let is_trait = match &store[*target] {
                    TypeRef::Path(p) => p.is_self_type(),
                    TypeRef::TypeParam(p) => Some(p.local_id()) == trait_self,
                    _ => false,
                };
                match is_trait {
                    true => bound.as_path(&store),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsafety {
    Safe,
    Unsafe,
    /// A lint.
    DeprecatedSafe2024,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFeatureIsSafeInTarget {
    No,
    Yes,
}

pub fn target_feature_is_safe_in_target(target: &TargetData) -> TargetFeatureIsSafeInTarget {
    match target.arch {
        target::Arch::Wasm32 | target::Arch::Wasm64 => TargetFeatureIsSafeInTarget::Yes,
        _ => TargetFeatureIsSafeInTarget::No,
    }
}

pub fn is_fn_unsafe_to_call(
    db: &dyn HirDatabase,
    func: FunctionId,
    caller_target_features: &TargetFeatures,
    call_edition: Edition,
    target_feature_is_safe: TargetFeatureIsSafeInTarget,
) -> Unsafety {
    let data = db.function_signature(func);
    if data.is_unsafe() {
        return Unsafety::Unsafe;
    }

    if data.has_target_feature() && target_feature_is_safe == TargetFeatureIsSafeInTarget::No {
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

    let loc = func.lookup(db);
    match loc.container {
        hir_def::ItemContainerId::ExternBlockId(block) => {
            let is_intrinsic_block = block.abi(db) == Some(sym::rust_dash_intrinsic);
            if is_intrinsic_block {
                // legacy intrinsics
                // extern "rust-intrinsic" intrinsics are unsafe unless they have the rustc_safe_intrinsic attribute
                if db.attrs(func.into()).by_key(sym::rustc_safe_intrinsic).exists() {
                    Unsafety::Safe
                } else {
                    Unsafety::Unsafe
                }
            } else {
                // Function in an `extern` block are always unsafe to call, except when
                // it is marked as `safe`.
                if data.is_safe() { Unsafety::Safe } else { Unsafety::Unsafe }
            }
        }
        _ => Unsafety::Safe,
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
            (e.enum_variants(db).variants[index.0].0, layout)
        }
        hir_def::layout::Variants::Multiple { tag, tag_encoding, variants, .. } => {
            let size = tag.size(target_data_layout).bytes_usize();
            let offset = layout.fields.offset(0).bytes_usize(); // The only field on enum variants is the tag field
            let tag = i128::from_le_bytes(pad16(&b[offset..offset + size], false));
            match tag_encoding {
                TagEncoding::Direct => {
                    let (var_idx, layout) =
                        variants.iter_enumerated().find_map(|(var_idx, v)| {
                            let def = e.enum_variants(db).variants[var_idx.0].0;
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
                    (e.enum_variants(db).variants[variant.0].0, &variants[variant])
                }
            }
        }
    };
    Some((var_id, var_layout))
}
