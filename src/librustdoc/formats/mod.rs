pub(crate) mod cache;
pub(crate) mod item_type;
pub(crate) mod renderer;

use rustc_hir::def_id::DefId;

pub(crate) use renderer::{run_format, FormatRenderer};

use crate::clean::{self, ItemId};

/// Specifies whether rendering directly implemented trait items or ones from a certain Deref
/// impl.
pub(crate) enum AssocItemRender<'a> {
    All,
    DerefFor { trait_: &'a clean::Path, type_: &'a clean::Type, deref_mut_: bool },
}

/// For different handling of associated items from the Deref target of a type rather than the type
/// itself.
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum RenderMode {
    Normal,
    ForDeref { mut_: bool },
}

/// Metadata about implementations for a type or trait.
#[derive(Clone, Debug)]
pub(crate) struct Impl {
    pub(crate) impl_item: clean::Item,
}

impl Impl {
    pub(crate) fn inner_impl(&self) -> &clean::Impl {
        match *self.impl_item.kind {
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
            ItemId::Primitive(_, _) => {
                panic!(
                    "Unexpected ItemId::Primitive in expect_def_id: {:?}",
                    self.impl_item.item_id
                )
            }
        }
    }
}
