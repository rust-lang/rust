//! Trace is a pretty niche data structure which is used when lowering a CST
//! into HIR.
//!
//! Lowering process calculates two bits of information:
//! * the lowered syntax itself
//! * a mapping between lowered syntax and original syntax
//!
//! Due to the way salsa works, the mapping is usually hot lava, as it contains
//! absolute offsets. The `Trace` structure (inspired, at least in name, by
//! Kotlin's `BindingTrace`) allows use the same code to compute both
//! projections.
use ra_arena::{map::ArenaMap, Arena, ArenaId, RawId};

pub(crate) struct Trace<ID: ArenaId, T, V> {
    arena: Option<Arena<ID, T>>,
    map: Option<ArenaMap<ID, V>>,
    len: u32,
}

impl<ID: ra_arena::ArenaId + Copy, T, V> Trace<ID, T, V> {
    pub(crate) fn new_for_arena() -> Trace<ID, T, V> {
        Trace { arena: Some(Arena::default()), map: None, len: 0 }
    }

    pub(crate) fn new_for_map() -> Trace<ID, T, V> {
        Trace { arena: None, map: Some(ArenaMap::default()), len: 0 }
    }

    pub(crate) fn alloc(&mut self, value: impl FnOnce() -> V, data: impl FnOnce() -> T) -> ID {
        let id = if let Some(arena) = &mut self.arena {
            arena.alloc(data())
        } else {
            let id = ID::from_raw(RawId::from(self.len));
            self.len += 1;
            id
        };

        if let Some(map) = &mut self.map {
            map.insert(id, value());
        }
        id
    }

    pub(crate) fn into_arena(mut self) -> Arena<ID, T> {
        self.arena.take().unwrap()
    }

    pub(crate) fn into_map(mut self) -> ArenaMap<ID, V> {
        self.map.take().unwrap()
    }
}
