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
use la_arena::{Arena, ArenaMap, Idx, RawIdx};

pub(crate) struct Trace<T, V> {
    arena: Option<Arena<T>>,
    map: Option<ArenaMap<Idx<T>, V>>,
    len: u32,
}

impl<T, V> Trace<T, V> {
    pub(crate) fn new_for_arena() -> Trace<T, V> {
        Trace { arena: Some(Arena::default()), map: None, len: 0 }
    }

    pub(crate) fn new_for_map() -> Trace<T, V> {
        Trace { arena: None, map: Some(ArenaMap::default()), len: 0 }
    }

    pub(crate) fn alloc(&mut self, value: impl FnOnce() -> V, data: impl FnOnce() -> T) -> Idx<T> {
        let id = if let Some(arena) = &mut self.arena {
            arena.alloc(data())
        } else {
            let id = Idx::<T>::from_raw(RawIdx::from(self.len));
            self.len += 1;
            id
        };

        if let Some(map) = &mut self.map {
            map.insert(id, value());
        }
        id
    }

    pub(crate) fn into_arena(mut self) -> Arena<T> {
        self.arena.take().unwrap()
    }

    pub(crate) fn into_map(mut self) -> ArenaMap<Idx<T>, V> {
        self.map.take().unwrap()
    }
}
