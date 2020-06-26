pub mod cache;
pub mod item_type;
pub mod renderer;

pub use renderer::{FormatRenderer, Renderer};

use rustc_span::def_id::DefId;

use crate::clean;
use crate::clean::types::GetDefId;

pub enum AssocItemRender<'a> {
    All,
    DerefFor { trait_: &'a clean::Type, type_: &'a clean::Type, deref_mut_: bool },
}

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
        match self.impl_item.inner {
            clean::ImplItem(ref impl_) => impl_,
            _ => panic!("non-impl item found in impl"),
        }
    }

    pub fn trait_did(&self) -> Option<DefId> {
        self.inner_impl().trait_.def_id()
    }
}
