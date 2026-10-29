use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;

#[global_allocator]
static ALLOCATOR: ArenaAllocator = ArenaAllocator::new();

/// Very simple allocator which never deallocates memory
///
/// Based on the example from
/// https://doc.rust-lang.org/stable/std/alloc/trait.GlobalAlloc.html
pub struct ArenaAllocator {
    arena: UnsafeCell<Arena>,
}

impl ArenaAllocator {
    pub const fn new() -> Self {
        Self { arena: UnsafeCell::new(Arena::new()) }
    }
}

/// Safe because we are singlethreaded
unsafe impl Sync for ArenaAllocator {}

unsafe impl GlobalAlloc for ArenaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let arena = &mut *self.arena.get();
        arena.alloc(layout)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

const ARENA_SIZE: usize = 64 * 1024; // more than enough

#[repr(C, align(4096))]
struct Arena {
    buf: [u8; ARENA_SIZE], // aligned at 4096
    allocated: usize,
}

impl Arena {
    pub const fn new() -> Self {
        Self { buf: [0x55; ARENA_SIZE], allocated: 0 }
    }

    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        if layout.align() > 4096 || layout.size() > ARENA_SIZE {
            return core::ptr::null_mut();
        }

        let align_minus_one = layout.align() - 1;
        let start = (self.allocated + align_minus_one) & !align_minus_one; // round up
        let new_cursor = start + layout.size();

        if new_cursor >= ARENA_SIZE {
            return core::ptr::null_mut();
        }

        self.allocated = new_cursor;
        self.buf.as_mut_ptr().add(start)
    }
}
