use ra_db::SourceRootId;

use crate::{
    hir::{SourceItemId, ModuleId},
};

use ra_db::{NumericId, LocationIntener};

macro_rules! impl_numeric_id {
    ($id:ident) => {
        impl NumericId for $id {
            fn from_u32(id: u32) -> Self {
                $id(id)
            }
            fn to_u32(self) -> u32 {
                self.0
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FnId(u32);
impl_numeric_id!(FnId);

impl FnId {
    pub(crate) fn from_loc(
        db: &impl AsRef<LocationIntener<SourceItemId, FnId>>,
        loc: &SourceItemId,
    ) -> FnId {
        db.as_ref().loc2id(loc)
    }
    pub(crate) fn loc(self, db: &impl AsRef<LocationIntener<SourceItemId, FnId>>) -> SourceItemId {
        db.as_ref().id2loc(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DefId(u32);
impl_numeric_id!(DefId);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DefLoc {
    Module {
        id: ModuleId,
        source_root: SourceRootId,
    },
    Item {
        source_item_id: SourceItemId,
    },
}

impl DefId {
    pub(crate) fn loc(self, db: &impl AsRef<LocationIntener<DefLoc, DefId>>) -> DefLoc {
        db.as_ref().id2loc(self)
    }
}

impl DefLoc {
    pub(crate) fn id(&self, db: &impl AsRef<LocationIntener<DefLoc, DefId>>) -> DefId {
        db.as_ref().loc2id(&self)
    }
}

#[derive(Debug, Default)]
pub(crate) struct IdMaps {
    pub(crate) fns: LocationIntener<SourceItemId, FnId>,
    pub(crate) defs: LocationIntener<DefLoc, DefId>,
}
