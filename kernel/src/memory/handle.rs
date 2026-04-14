use super::arena::ArenaId;
use core::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictHandle {
    pub arena_id: ArenaId,
    pub generation: u64,
    pub offset: u32,
    pub len: u32,
}

impl EvictHandle {
    pub fn new(arena_id: ArenaId, generation: u64, offset: u32, len: u32) -> Self {
        Self {
            arena_id,
            generation,
            offset,
            len,
        }
    }

    /// Resolve the handle to a pointer.
    /// Returns `None` if the arena has been evicted (generation mismatch) or doesn't exist.
    pub fn resolve(&self) -> Option<NonNull<u8>> {
        // This requires access to the global ArenaHeap.
        // We will need to expose a way to query the global heap.
        // For now, this is a placeholder. The actual resolution logic
        // will likely need to be a method on `ArenaHeap` or a global helper function
        // that locks the global allocator.

        // Option 1: Pass the heap to resolve.
        // Option 2: Use a global accessor (e.g. `kernel::memory::arena::resolve(self)`)

        // Implementation will be done in `kheap.rs` or `mod.rs` where we have access to the global lock.
        // For now, let's leave this struct definition here.
        None
    }
}
