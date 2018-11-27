use parking_lot::Mutex;

use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::{
    hir::{SourceItemId, ModuleId},
    input::SourceRootId,
};

/// There are two principle ways to refer to things:
///   - by their locatinon (module in foo/bar/baz.rs at line 42)
///   - by their numeric id (module `ModuleId(42)`)
///
/// The first one is more powerful (you can actually find the thing in question
/// by id), but the second one is so much more compact.
///
/// `Loc2IdMap` allows us to have a cake an eat it as well: by maintaining a
/// bidirectional mapping between positional and numeric ids, we can use compact
/// representation wich still allows us to get the actual item
#[derive(Debug)]
struct Loc2IdMap<LOC, ID>
where
    ID: NumericId,
    LOC: Clone + Eq + Hash,
{
    loc2id: FxHashMap<LOC, ID>,
    id2loc: FxHashMap<ID, LOC>,
}

impl<LOC, ID> Default for Loc2IdMap<LOC, ID>
where
    ID: NumericId,
    LOC: Clone + Eq + Hash,
{
    fn default() -> Self {
        Loc2IdMap {
            loc2id: FxHashMap::default(),
            id2loc: FxHashMap::default(),
        }
    }
}

impl<LOC, ID> Loc2IdMap<LOC, ID>
where
    ID: NumericId,
    LOC: Clone + Eq + Hash,
{
    pub fn loc2id(&mut self, loc: &LOC) -> ID {
        match self.loc2id.get(loc) {
            Some(id) => return id.clone(),
            None => (),
        }
        let id = self.loc2id.len();
        assert!(id < u32::max_value() as usize);
        let id = ID::from_u32(id as u32);
        self.loc2id.insert(loc.clone(), id.clone());
        self.id2loc.insert(id.clone(), loc.clone());
        id
    }

    pub fn id2loc(&self, id: ID) -> LOC {
        self.id2loc[&id].clone()
    }
}

pub(crate) trait NumericId: Clone + Eq + Hash {
    fn from_u32(id: u32) -> Self;
    fn to_u32(self) -> u32;
}

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

#[derive(Debug)]
pub(crate) struct LocationIntener<LOC, ID>
where
    ID: NumericId,
    LOC: Clone + Eq + Hash,
{
    map: Mutex<Loc2IdMap<LOC, ID>>,
}

impl<LOC, ID> Default for LocationIntener<LOC, ID>
where
    ID: NumericId,
    LOC: Clone + Eq + Hash,
{
    fn default() -> Self {
        LocationIntener {
            map: Default::default(),
        }
    }
}

impl<LOC, ID> LocationIntener<LOC, ID>
where
    ID: NumericId,
    LOC: Clone + Eq + Hash,
{
    fn loc2id(&self, loc: &LOC) -> ID {
        self.map.lock().loc2id(loc)
    }
    fn id2loc(&self, id: ID) -> LOC {
        self.map.lock().id2loc(id)
    }
}
