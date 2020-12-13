crate mod cache;
crate mod item_type;
crate mod renderer;

crate use renderer::{run_format, FormatRenderer};

use rustc_span::def_id::DefId;

use crate::clean;
use crate::clean::types::GetDefId;

/// Specifies whether rendering directly implemented trait items or ones from a certain Deref
/// impl.
crate enum AssocItemRender<'a> {
    All,
    DerefFor { trait_: &'a clean::Type, type_: &'a clean::Type, deref_mut_: bool },
}

/// For different handling of associated items from the Deref target of a type rather than the type
/// itself.
#[derive(Copy, Clone, PartialEq)]
crate enum RenderMode {
    Normal,
    ForDeref { mut_: bool },
}

/// Metadata about implementations for a type or trait.
#[derive(Clone, Debug)]
crate struct Impl {
    crate impl_item: clean::Item,
}

impl Impl {
    crate fn inner_impl(&self) -> &clean::Impl {
        match *self.impl_item.kind {
            clean::ImplItem(ref impl_) => impl_,
            _ => panic!("non-impl item found in impl"),
        }
    }

    crate fn trait_did(&self) -> Option<DefId> {
        self.inner_impl().trait_.def_id()
    }
}
