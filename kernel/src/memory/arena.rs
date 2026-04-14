use core::alloc::Layout;
use core::ptr::NonNull;

pub const MAX_ARENAS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArenaId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct Mark {
    pub arena_id: ArenaId,
    pub cursor: usize,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ArenaFlags: u8 {
        const PINNED = 0x01;
        const EVICTABLE = 0x02;
    }
}

#[derive(Clone)]
pub struct Arena {
    pub base: u64,     // Virtual Address
    pub size: usize,   // Total size in bytes
    pub cursor: usize, // Current offset
    pub flags: ArenaFlags,
    pub generation: u64,
    pub tag: &'static str,

    // Intrusive list links
    pub next: Option<ArenaId>,
    pub prev: Option<ArenaId>,
}

impl Arena {
    pub fn new(
        base: u64,
        size: usize,
        flags: ArenaFlags,
        generation: u64,
        tag: &'static str,
    ) -> Self {
        Self {
            base,
            size,
            cursor: 0,
            flags,
            generation,
            tag,
            next: None,
            prev: None,
        }
    }

    pub fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
        let current = self.base + self.cursor as u64;
        let align = layout.align() as u64;
        let align_offset = (align - (current % align)) % align;

        let new_cursor = self.cursor + align_offset as usize + layout.size();

        if new_cursor > self.size {
            return Err(());
        }

        let ptr = (self.base + self.cursor as u64 + align_offset as u64) as *mut u8;
        self.cursor = new_cursor;

        // if layout.size() > 1024 {
        //      crate::kprintln!("Arena[{}] Alloc: size={} ptr={:p}", self.tag, layout.size(), ptr);
        // }

        unsafe { Ok(NonNull::new_unchecked(ptr)) }
    }

    pub fn used(&self) -> usize {
        self.cursor
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AllocatorStats {
    pub total_pinned_bytes: usize,
    pub total_evictable_bytes: usize,
    pub eviction_count: usize,
    pub bytes_freed_by_eviction: usize,
    pub largest_alloc_request: usize,
}

pub struct ArenaHeap {
    arenas: [Option<Arena>; MAX_ARENAS],
    // Pinned list (append only, alloc from tail)
    pinned_tail: Option<ArenaId>,

    // Evictable list (FIFO: push tail, pop head)
    evictable_head: Option<ArenaId>,
    evictable_tail: Option<ArenaId>,

    stats: AllocatorStats,
    // Callbacks removed for now to avoid Box allocation
}

impl ArenaHeap {
    pub const fn new() -> Self {
        Self {
            arenas: [const { None }; MAX_ARENAS],
            pinned_tail: None,
            evictable_head: None,
            evictable_tail: None,
            stats: AllocatorStats {
                total_pinned_bytes: 0,
                total_evictable_bytes: 0,
                eviction_count: 0,
                bytes_freed_by_eviction: 0,
                largest_alloc_request: 0,
            },
        }
    }

    /// Add a pre-existing memory region as an arena
    pub fn add_arena(&mut self, mut arena: Arena) -> ArenaId {
        crate::kinfo!(
            "ArenaHeap: Adding arena '{}' base={:x} size={} flags={:?}",
            arena.tag,
            arena.base,
            arena.size,
            arena.flags
        );

        // Find free slot
        let idx = self
            .arenas
            .iter()
            .position(|slot| slot.is_none())
            .expect("ArenaHeap: Max arenas limit reached!"); // Panic if full (fix later?)

        let id = ArenaId(idx as u32);
        let is_pinned = arena.flags.contains(ArenaFlags::PINNED);

        if is_pinned {
            // Add to pinned list (tail)
            arena.prev = self.pinned_tail;
            arena.next = None;

            if let Some(tail_id) = self.pinned_tail {
                if let Some(Some(tail)) = self.arenas.get_mut(tail_id.0 as usize) {
                    tail.next = Some(id);
                }
            }
            self.pinned_tail = Some(id);
        } else {
            // Add to evictable list (tail)
            arena.prev = self.evictable_tail;
            arena.next = None;

            if let Some(tail_id) = self.evictable_tail {
                if let Some(Some(tail)) = self.arenas.get_mut(tail_id.0 as usize) {
                    tail.next = Some(id);
                }
            }
            self.evictable_tail = Some(id);
            if self.evictable_head.is_none() {
                self.evictable_head = Some(id);
            }
        }

        self.arenas[idx] = Some(arena);
        id
    }

    pub fn alloc_pinned(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
        if layout.size() > self.stats.largest_alloc_request {
            self.stats.largest_alloc_request = layout.size();
        }

        // Try allocation in the latest pinned arena (tail)
        if let Some(id) = self.pinned_tail {
            if let Some(Some(arena)) = self.arenas.get_mut(id.0 as usize) {
                if let Ok(ptr) = arena.alloc(layout) {
                    self.stats.total_pinned_bytes += layout.size();
                    return Ok(ptr);
                }
            }
        }

        Err(())
    }

    pub fn alloc_evictable(
        &mut self,
        layout: Layout,
    ) -> Result<(ArenaId, u64, u32, NonNull<u8>), ()> {
        if layout.size() > self.stats.largest_alloc_request {
            self.stats.largest_alloc_request = layout.size();
        }

        // Try allocation in the latest evictable arena (tail)
        if let Some(id) = self.evictable_tail {
            if let Some(Some(arena)) = self.arenas.get_mut(id.0 as usize) {
                if let Ok(ptr) = arena.alloc(layout) {
                    self.stats.total_evictable_bytes += layout.size();
                    let offset = (ptr.as_ptr() as u64 - arena.base) as u32;
                    return Ok((id, arena.generation, offset, ptr));
                }
            }
        }

        Err(())
    }

    /// Evict the oldest evictable arena. Returns the arena so it can be recycled or freed.
    pub fn evict_oldest(&mut self) -> Option<Arena> {
        let id_to_evict = self.evictable_head?;

        // Remove from list
        let mut arena = self.arenas[id_to_evict.0 as usize].take()?;

        crate::kinfo!(
            "ArenaHeap: Evicting arena '{}' id={:?} used={}",
            arena.tag,
            id_to_evict,
            arena.used()
        );

        // Update list pointers
        let next_id = arena.next;
        if let Some(next_id) = next_id {
            if let Some(Some(next_arena)) = self.arenas.get_mut(next_id.0 as usize) {
                next_arena.prev = None;
            }
        }

        self.evictable_head = next_id;
        if self.evictable_head.is_none() {
            self.evictable_tail = None;
        }

        // Update stats
        self.stats.eviction_count += 1;
        self.stats.bytes_freed_by_eviction += arena.size;
        self.stats.total_evictable_bytes = self
            .stats
            .total_evictable_bytes
            .saturating_sub(arena.used());

        // Unlink from the arena itself (clean state)
        arena.next = None;
        arena.prev = None;

        Some(arena)
    }

    pub fn get_arena(&self, id: ArenaId) -> Option<&Arena> {
        self.arenas.get(id.0 as usize)?.as_ref()
    }

    pub fn mark_current(&self) -> Option<Mark> {
        let id = self.pinned_tail?;
        let arena = self.arenas.get(id.0 as usize)?.as_ref()?;
        Some(Mark {
            arena_id: id,
            cursor: arena.cursor,
        })
    }

    pub fn rewind(&mut self, mark: Mark) {
        if let Some(Some(arena)) = self.arenas.get_mut(mark.arena_id.0 as usize) {
            if mark.cursor < arena.cursor {
                arena.cursor = mark.cursor;
            }
        }
    }

    pub fn stats(&self) -> AllocatorStats {
        self.stats
    }
}

// -----------------------------------------------------------------------
// Public Global API
// -----------------------------------------------------------------------

pub fn alloc_evictable<R: crate::BootRuntime>(
    layout: Layout,
) -> Result<crate::memory::handle::EvictHandle, ()> {
    crate::memory::kheap::kernel_heap()
        .lock()
        .alloc_evictable::<R>(layout)
}

pub fn evict_until<R: crate::BootRuntime>(bytes_needed: usize) -> usize {
    crate::memory::kheap::kernel_heap()
        .lock()
        .evict_until::<R>(bytes_needed)
}

pub fn mark() -> Option<Mark> {
    crate::memory::kheap::kernel_heap()
        .lock()
        .arena_system
        .mark_current()
}

pub fn rewind(mark: Mark) {
    crate::memory::kheap::kernel_heap()
        .lock()
        .arena_system
        .rewind(mark)
}

pub fn heap_status() {
    let heap = crate::memory::kheap::kernel_heap().lock();
    let stats = heap.stats();
    crate::kinfo!("Heap Status:");
    crate::kinfo!("  Pinned:    {} bytes", stats.total_pinned_bytes);
    crate::kinfo!("  Evictable: {} bytes", stats.total_evictable_bytes);
    crate::kinfo!(
        "  Evictions: {} (freed {} bytes)",
        stats.eviction_count,
        stats.bytes_freed_by_eviction
    );
    crate::kinfo!("  Max Req:   {} bytes", stats.largest_alloc_request);
}

// Ensure tests compile (updated for new API)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_alloc() {
        let mut heap = ArenaHeap::new();
        let base = 0x1000;
        let size = 1000;
        let arena = Arena::new(base, size, ArenaFlags::PINNED, 0, "test");
        heap.add_arena(arena);

        let layout = Layout::from_size_align(10, 1).unwrap();
        let ptr = heap.alloc_pinned(layout).expect("Should alloc");
        assert_eq!(ptr.as_ptr() as u64, base);
    }

    #[test]
    fn test_evictable_alloc_and_eviction() {
        let mut heap = ArenaHeap::new();
        let base1 = 0x2000;
        let size1 = 1000;
        let arena1 = Arena::new(base1, size1, ArenaFlags::EVICTABLE, 1, "test_evict1");
        heap.add_arena(arena1);

        let base2 = 0x3000;
        let size2 = 1000;
        let arena2 = Arena::new(base2, size2, ArenaFlags::EVICTABLE, 2, "test_evict2");
        heap.add_arena(arena2);

        let layout = Layout::from_size_align(20, 1).unwrap();

        // Allocate should go to the tail (arena2)
        let (id, generation, offset, ptr) = heap
            .alloc_evictable(layout)
            .expect("Should alloc evictable");
        assert_eq!(generation, 2);
        assert_eq!(ptr.as_ptr() as u64, base2);
        assert_eq!(offset, 0);

        let stats = heap.stats();
        assert_eq!(stats.total_evictable_bytes, 20);

        // Evict the oldest evictable arena (arena1)
        let evicted = heap.evict_oldest().expect("Should evict arena");
        assert_eq!(evicted.tag, "test_evict1");
        assert_eq!(evicted.base, base1);

        let stats_after = heap.stats();
        assert_eq!(stats_after.eviction_count, 1);
        assert_eq!(stats_after.bytes_freed_by_eviction, size1);

        // Evict the remaining evictable arena (arena2)
        let evicted2 = heap.evict_oldest().expect("Should evict second arena");
        assert_eq!(evicted2.tag, "test_evict2");
        assert_eq!(evicted2.base, base2);

        // No more arenas to evict
        assert!(heap.evict_oldest().is_none());
    }
}
