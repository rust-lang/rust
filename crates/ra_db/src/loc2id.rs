use std::{panic, hash::Hash};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use ra_arena::{Arena, ArenaId};

/// There are two principle ways to refer to things:
///   - by their location (module in foo/bar/baz.rs at line 42)
///   - by their numeric id (module `ModuleId(42)`)
///
/// The first one is more powerful (you can actually find the thing in question
/// by id), but the second one is so much more compact.
///
/// `Loc2IdMap` allows us to have a cake an eat it as well: by maintaining a
/// bidirectional mapping between positional and numeric ids, we can use compact
/// representation which still allows us to get the actual item.
#[derive(Debug)]
struct Loc2IdMap<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
{
    id2loc: Arena<ID, LOC>,
    loc2id: FxHashMap<LOC, ID>,
}

impl<LOC, ID> Default for Loc2IdMap<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
{
    fn default() -> Self {
        Loc2IdMap { id2loc: Arena::default(), loc2id: FxHashMap::default() }
    }
}

impl<LOC, ID> Loc2IdMap<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
{
    pub fn len(&self) -> usize {
        self.id2loc.len()
    }

    pub fn loc2id(&mut self, loc: &LOC) -> ID {
        match self.loc2id.get(loc) {
            Some(id) => return id.clone(),
            None => (),
        }
        let id = self.id2loc.alloc(loc.clone());
        self.loc2id.insert(loc.clone(), id.clone());
        id
    }

    pub fn id2loc(&self, id: ID) -> LOC {
        self.id2loc[id].clone()
    }
}

#[derive(Debug)]
pub struct LocationIntener<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
{
    map: Mutex<Loc2IdMap<LOC, ID>>,
}

impl<LOC, ID> panic::RefUnwindSafe for LocationIntener<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
    ID: panic::RefUnwindSafe,
    LOC: panic::RefUnwindSafe,
{
}

impl<LOC, ID> Default for LocationIntener<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
{
    fn default() -> Self {
        LocationIntener { map: Default::default() }
    }
}

impl<LOC, ID> LocationIntener<LOC, ID>
where
    ID: ArenaId + Clone,
    LOC: Clone + Eq + Hash,
{
    pub fn len(&self) -> usize {
        self.map.lock().len()
    }
    pub fn loc2id(&self, loc: &LOC) -> ID {
        self.map.lock().loc2id(loc)
    }
    pub fn id2loc(&self, id: ID) -> LOC {
        self.map.lock().id2loc(id)
    }
}
