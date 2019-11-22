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
    for_arena: bool,
    arena: Arena<ID, T>,
    map: ArenaMap<ID, V>,
    len: u32,
}

impl<ID: ra_arena::ArenaId, T, V> Trace<ID, T, V> {
    pub(crate) fn new_for_arena() -> Trace<ID, T, V> {
        Trace { for_arena: true, arena: Arena::default(), map: ArenaMap::default(), len: 0 }
    }

    pub(crate) fn new_for_map() -> Trace<ID, T, V> {
        Trace { for_arena: false, arena: Arena::default(), map: ArenaMap::default(), len: 0 }
    }

    pub(crate) fn alloc(&mut self, value: impl Fn() -> V, data: impl Fn() -> T) {
        if self.for_arena {
            self.arena.alloc(data());
        } else {
            let id = ID::from_raw(RawId::from(self.len));
            self.len += 1;
            self.map.insert(id, value());
        }
    }

    pub(crate) fn into_arena(self) -> Arena<ID, T> {
        assert!(self.for_arena);
        self.arena
    }

    pub(crate) fn into_map(self) -> ArenaMap<ID, V> {
        assert!(!self.for_arena);
        self.map
    }
}
