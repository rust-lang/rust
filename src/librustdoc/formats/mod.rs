pub(crate) mod cache;
pub(crate) mod item_type;
pub(crate) mod renderer;

pub(crate) use renderer::{FormatRenderer, run_format};
use rustc_hir::def_id::DefId;

use crate::clean::{self, ItemId};
use crate::html::render::Context;

/// Metadata about implementations for a type or trait.
#[derive(Clone, Debug)]
pub(crate) struct Impl {
    pub(crate) impl_item: clean::Item,
}

impl Impl {
    pub(crate) fn inner_impl(&self) -> &clean::Impl {
        match self.impl_item.kind {
            clean::ImplItem(ref impl_) => impl_,
            _ => panic!("non-impl item found in impl"),
        }
    }

    pub(crate) fn trait_did(&self) -> Option<DefId> {
        self.inner_impl().trait_.as_ref().map(|t| t.def_id())
    }

    /// This function is used to extract a `DefId` to be used as a key for the `Cache::impls` field.
    ///
    /// It allows to prevent having duplicated implementations showing up (the biggest issue was
    /// with blanket impls).
    ///
    /// It panics if `self` is a `ItemId::Primitive`.
    pub(crate) fn def_id(&self) -> DefId {
        match self.impl_item.item_id {
            ItemId::Blanket { impl_id, .. } => impl_id,
            ItemId::Auto { trait_, .. } => trait_,
            ItemId::DefId(def_id) => def_id,
        }
    }

    // Returns true if this is an implementation on a "local" type, meaning:
    // the type is in the current crate, or the type and the trait are both
    // re-exported by the current crate.
    pub(crate) fn is_on_local_type(&self, cx: &Context<'_>) -> bool {
        let cache = cx.cache();
        let for_type = &self.inner_impl().for_;
        if let Some(for_type_did) = for_type.def_id(cache) {
            // The "for" type is local if it's in the paths for the current crate.
            if cache.paths.contains_key(&for_type_did) {
                return true;
            }
            if let Some(trait_did) = self.trait_did() {
                // The "for" type and the trait are from the same crate. That could
                // be different from the current crate, for instance when both were
                // re-exported from some other crate. But they are local with respect to
                // each other.
                if for_type_did.krate == trait_did.krate {
                    return true;
                }
                // Hack: many traits and types in std are re-exported from
                // core or alloc. In general, rustdoc is capable of recognizing
                // these implementations as being on local types. However, in at
                // least one case (https://github.com/rust-lang/rust/issues/97610),
                // rustdoc gets confused and labels an implementation as being on
                // a foreign type. To make sure that confusion doesn't pass on to
                // the reader, consider all implementations in std, core, and alloc
                // to be on local types.
                let crate_name = cx.tcx().crate_name(trait_did.krate);
                if matches!(crate_name.as_str(), "std" | "core" | "alloc") {
                    return true;
                }
            }
            return false;
        };
        true
    }
}
