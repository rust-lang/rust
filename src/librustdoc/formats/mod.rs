pub mod cache;
pub mod item_type;
pub mod renderer;

pub use renderer::{run_format, FormatRenderer};

use rustc_span::def_id::DefId;

use crate::clean;
use crate::clean::types::GetDefId;

/// Specifies whether rendering directly implemented trait items or ones from a certain Deref
/// impl.
pub enum AssocItemRender<'a> {
    All,
    DerefFor { trait_: &'a clean::Type, type_: &'a clean::Type, deref_mut_: bool },
}

/// For different handling of associated items from the Deref target of a type rather than the type
/// itself.
#[derive(Copy, Clone, PartialEq)]
pub enum RenderMode {
    Normal,
    ForDeref { mut_: bool },
}

/// Metadata about implementations for a type or trait.
#[derive(Clone, Debug)]
pub struct Impl {
    pub impl_item: clean::Item,
}

impl Impl {
    pub fn inner_impl(&self) -> &clean::Impl {
        match self.impl_item.kind {
            clean::ImplItem(ref impl_) => impl_,
            _ => panic!("non-impl item found in impl"),
        }
    }

    pub fn trait_did(&self) -> Option<DefId> {
        self.inner_impl().trait_.def_id()
    }
}
