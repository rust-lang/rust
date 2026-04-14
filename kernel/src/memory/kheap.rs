use super::arena::{AllocatorStats, Arena, ArenaFlags, ArenaHeap};
use super::handle::EvictHandle;
use crate::{BootRuntime, BootTasking, MapKind, MapPerms, runtime};
use core::alloc::Layout;
use core::ptr::NonNull;

pub struct KernelHeap {
    /// Start of the kernel dynamic heap virtual address range
    #[allow(dead_code)]
    heap_base: u64,
    /// Next available virtual address for a new arena
    next_va: u64,
    /// The manager of all arenas
    pub arena_system: ArenaHeap,
}

impl KernelHeap {
    pub const fn new() -> Self {
        Self {
            // Start at 0xffffffffb0000000 (same as before)
            heap_base: 0xffffffffb0000000,
            next_va: 0xffffffffb0000000,
            arena_system: ArenaHeap::new(),
        }
    }

    pub fn init(&mut self, _size_mb: usize) {
        // Initialization if needed
    }

    /// Add a new pinned arena of the given number of pages.
    /// This is used by the Global Allocator when it runs out of space.
    pub fn expand_pinned<R: BootRuntime>(&mut self, pages: usize) -> Result<(), ()> {
        let (base, size) = self.map_new_region::<R>(pages)?;

        let arena = Arena::new(
            base,
            size,
            ArenaFlags::PINNED,
            0, // Generation 0 for pinned
            "global.pinned",
        );

        self.arena_system.add_arena(arena);
        Ok(())
    }

    /// Helper to map a new contiguous region of `pages`
    fn map_new_region<R: BootRuntime>(&mut self, pages: usize) -> Result<(u64, usize), ()> {
        let rt = runtime::<R>();
        let tasking = rt.tasking();
        let aspace = tasking.active_address_space();
        let size = pages * 4096;
        let base_va = self.next_va;

        // Ensure we don't overflow (basic check, refined later for strict limits)
        if base_va.checked_add(size as u64).is_none() {
            return Err(());
        }

        // We allocate pages one by one to ensure we can fulfill the request
        // Ideally we'd alloc_contiguous but map_page handles 4K chunks.
        // For an arena, virtual contiguity is required. Physical contiguity is NOT required
        // because we access via VA.

        for i in 0..pages {
            let offset = (i * 4096) as u64;
            let va = base_va + offset;

            // Allocate a physical frame
            let phys = super::alloc_frame().ok_or(())?;

            tasking.map_page(
                aspace,
                va,
                phys,
                MapPerms {
                    read: true,
                    write: true,
                    user: false,
                    exec: false,
                    kind: MapKind::Normal,
                },
                MapKind::Normal,
                &KernelFrameHook,
            )?;
        }

        self.next_va += size as u64;

        // NEW: Broadcast TLB shootdown to other CPUs - essential for SMP heap consistency
        rt.tlb_shootdown_broadcast();

        Ok((base_va, size))
    }

    /// Add a new evictable arena of the given number of pages.
    pub fn expand_evictable<R: BootRuntime>(&mut self, pages: usize) -> Result<(), ()> {
        let (base, size) = self.map_new_region::<R>(pages)?;

        let arena = Arena::new(
            base,
            size,
            ArenaFlags::EVICTABLE,
            1, // Start at generation 1
            "evictable.dynamic",
        );

        self.arena_system.add_arena(arena);
        Ok(())
    }

    /// Reserve a contiguous region of virtual memory mapped to physical frames.
    /// Used by the global allocator initialization.
    pub fn reserve_region<R: BootRuntime>(&mut self, pages: usize) -> Result<(u64, usize), ()> {
        self.map_new_region::<R>(pages)
    }

    pub fn alloc_pinned(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
        self.arena_system.alloc_pinned(layout)
    }

    pub fn alloc_evictable<R: BootRuntime>(&mut self, layout: Layout) -> Result<EvictHandle, ()> {
        // 1. Try allocation in existing available arena
        if let Ok((id, generation, off, _)) = self.arena_system.alloc_evictable(layout) {
            return Ok(EvictHandle::new(id, generation, off, layout.size() as u32));
        }

        // 2. Try to expand (add new arena).
        // Default to a sensible size, e.g. 256KB (64 pages) or larger if requested?
        // Let's ensure at least enough for the layout + overhead.
        // For simplicity, fixed 256KB chunks for now.
        let needed_pages = (layout.size() + 4095) / 4096;
        let expand_pages = core::cmp::max(64, needed_pages);

        if self.expand_evictable::<R>(expand_pages).is_ok() {
            // Retry alloc
            if let Ok((id, generation, off, _)) = self.arena_system.alloc_evictable(layout) {
                return Ok(EvictHandle::new(id, generation, off, layout.size() as u32));
            }
        }

        // 3. Eviction / Recycling loop
        // If expansion failed (OOM) or wasn't enough, try to evict old arenas to make space.
        // In a real scenario, we might want to unmap pages and return to frame allocator
        // to handle "Physical OOM".
        // But here, we can also just reuse the VA/PA of the evicted arena!

        while let Some(mut arena) = self.arena_system.evict_oldest() {
            // Recycle this arena:
            // 1. Reset cursor
            arena.cursor = 0;
            // 2. Increment generation (invalidates old handles)
            arena.generation += 1;

            // 3. Add back as "new" arena
            self.arena_system.add_arena(arena);

            // 4. Retry allocation
            if let Ok((id, generation, off, _)) = self.arena_system.alloc_evictable(layout) {
                return Ok(EvictHandle::new(id, generation, off, layout.size() as u32));
            }
            // If still fails (maybe arena too small?), loop continues to evict next oldest.
            // Note: If request size > arena size, this loop might evict everything and still fail.
        }

        Err(())
    }

    /// Evict arenas until `bytes_needed` are freed (returned to system).
    /// Returns total bytes freed.
    pub fn evict_until<R: BootRuntime>(&mut self, bytes_needed: usize) -> usize {
        let mut freed = 0;
        while freed < bytes_needed {
            // Get oldest arena
            if let Some(arena) = self.arena_system.evict_oldest() {
                let size = arena.size;
                freed += size;
                self.unmap_arena::<R>(arena);
            } else {
                break;
            }
        }
        freed
    }

    fn unmap_arena<R: BootRuntime>(&mut self, arena: Arena) {
        let rt = runtime::<R>();
        let tasking = rt.tasking();
        let aspace = tasking.active_address_space();

        // We assume arena.base and (base + size) are page aligned
        let pages = arena.size / 4096;

        for i in 0..pages {
            let va = arena.base + (i * 4096) as u64;
            // Unmap page
            if let Ok(Some(phys)) = tasking.unmap_page(aspace, va) {
                // Free frame
                super::FRAME_ALLOCATOR.with_lock(|a| a.mark_free_range(phys, phys + 4096));
            }
        }
    }

    pub fn stats(&self) -> AllocatorStats {
        self.arena_system.stats()
    }
}

pub struct KernelFrameHook;
impl crate::FrameAllocatorHook for KernelFrameHook {
    fn alloc_frame(&self) -> Option<u64> {
        super::alloc_frame()
    }
}

pub fn kernel_heap() -> &'static spin::Mutex<KernelHeap> {
    static HEAP: spin::Mutex<KernelHeap> = spin::Mutex::new(KernelHeap::new());
    &HEAP
}
