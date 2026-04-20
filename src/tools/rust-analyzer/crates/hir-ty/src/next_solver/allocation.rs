use std::{fmt, hash::Hash};

use intern::{Interned, InternedRef, impl_internable};
use macros::GenericTypeVisitable;
use rustc_type_ir::GenericTypeVisitable;

use crate::{
    MemoryMap,
    next_solver::{Ty, impl_stored_interned},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Allocation<'db> {
    interned: InternedRef<'db, AllocationInterned>,
}

impl<'db> Allocation<'db> {
    pub fn new(data: AllocationData<'db>) -> Self {
        let data =
            unsafe { std::mem::transmute::<AllocationData<'db>, AllocationData<'static>>(data) };
        Self { interned: Interned::new_gc(AllocationInterned(data)) }
    }
}

impl<'db> std::ops::Deref for Allocation<'db> {
    type Target = AllocationData<'db>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        let inner = &self.interned.0;
        unsafe { std::mem::transmute::<&AllocationData<'static>, &AllocationData<'db>>(inner) }
    }
}

impl<'db, V: super::WorldExposer> GenericTypeVisitable<V> for Allocation<'db> {
    fn generic_visit_with(&self, visitor: &mut V) {
        if visitor.on_interned(self.interned).is_continue() {
            (**self).generic_visit_with(visitor);
        }
    }
}

impl fmt::Debug for Allocation<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let AllocationData { ty, memory, memory_map } = &**self;
        f.debug_struct("Allocation")
            .field("ty", ty)
            .field("memory", memory)
            .field("memory_map", memory_map)
            .finish()
    }
}

#[derive(PartialEq, Eq, Hash, GenericTypeVisitable)]
pub(super) struct AllocationInterned(AllocationData<'static>);

#[derive(Debug, PartialEq, Eq, GenericTypeVisitable)]
pub struct AllocationData<'db> {
    pub ty: Ty<'db>,
    pub memory: Box<[u8]>,
    pub memory_map: MemoryMap<'db>,
}

impl<'db> Hash for AllocationData<'db> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let Self { ty, memory, memory_map: _ } = self;
        ty.hash(state);
        memory.hash(state);
    }
}

impl_internable!(gc; AllocationInterned);
impl_stored_interned!(AllocationInterned, Allocation, StoredAllocation);
