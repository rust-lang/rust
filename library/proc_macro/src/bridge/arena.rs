//! A minimal arena allocator inspired by `rustc_arena::DroplessArena`.
//!
//! This is unfortunately a minimal re-implementation rather than a dependency
//! as it is difficult to depend on crates from within `proc_macro`, due to it
//! being built at the same time as `std`.

use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::{cmp, ptr, slice, str};

// The arenas start with PAGE-sized chunks, and then each new chunk is twice as
// big as its predecessor, up until we reach HUGE_PAGE-sized chunks, whereupon
// we stop growing. This scales well, from arenas that are barely used up to
// arenas that are used for 100s of MiBs. Note also that the chosen sizes match
// the usual sizes of pages and huge pages on Linux.
const PAGE: usize = 4096;
const HUGE_PAGE: usize = 2 * 1024 * 1024;

/// A minimal arena allocator inspired by `rustc_arena::DroplessArena`.
///
/// This is unfortunately a complete re-implementation rather than a dependency
/// as it is difficult to depend on crates from within `proc_macro`, due to it
/// being built at the same time as `std`.
///
/// This arena doesn't have support for allocating anything other than byte
/// slices, as that is all that is necessary.
pub(crate) struct Arena {
    start: Cell<*mut MaybeUninit<u8>>,
    end: Cell<*mut MaybeUninit<u8>>,
    chunks: RefCell<Vec<Box<[MaybeUninit<u8>]>>>,
}

impl Arena {
    pub(crate) fn new() -> Self {
        Arena {
            start: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: RefCell::new(Vec::new()),
        }
    }

    /// Add a new chunk with at least `additional` free bytes.
    #[inline(never)]
    #[cold]
    fn grow(&self, additional: usize) {
        let mut chunks = self.chunks.borrow_mut();
        let mut new_cap;
        if let Some(last_chunk) = chunks.last_mut() {
            // If the previous chunk's len is less than HUGE_PAGE
            // bytes, then this chunk will be least double the previous
            // chunk's size.
            new_cap = last_chunk.len().min(HUGE_PAGE / 2);
            new_cap *= 2;
        } else {
            new_cap = PAGE;
        }
        // Also ensure that this chunk can fit `additional`.
        new_cap = cmp::max(additional, new_cap);

        let mut chunk = Box::new_uninit_slice(new_cap);
        let Range { start, end } = chunk.as_mut_ptr_range();
        self.start.set(start);
        self.end.set(end);
        chunks.push(chunk);
    }

    /// Allocates a byte slice with specified size from the current memory
    /// chunk. Returns `None` if there is no free space left to satisfy the
    /// request.
    fn alloc_raw_without_grow(&self, bytes: usize) -> Option<&mut [MaybeUninit<u8>]> {
        let start = self.start.get().addr();
        let old_end = self.end.get();
        let end = old_end.addr();

        let new_end = end.checked_sub(bytes)?;
        if start <= new_end {
            let new_end = old_end.with_addr(new_end);
            self.end.set(new_end);
            // SAFETY: `bytes` bytes starting at `new_end` were just reserved.
            Some(unsafe { slice::from_raw_parts_mut(new_end, bytes) })
        } else {
            None
        }
    }

    fn alloc_raw(&self, bytes: usize) -> &mut [MaybeUninit<u8>] {
        if bytes == 0 {
            return &mut [];
        }

        loop {
            if let Some(a) = self.alloc_raw_without_grow(bytes) {
                break a;
            }
            // No free space left. Allocate a new chunk to satisfy the request.
            // On failure the grow will panic or abort.
            self.grow(bytes);
        }
    }

    #[allow(clippy::mut_from_ref)] // arena allocator
    pub(crate) fn alloc_str<'a>(&'a self, string: &str) -> &'a mut str {
        let alloc = self.alloc_raw(string.len());
        let bytes = alloc.write_copy_of_slice(string.as_bytes());

        // SAFETY: we convert from `&str` to `&[u8]`, clone it into the arena,
        // and immediately convert the clone back to `&str`.
        unsafe { str::from_utf8_unchecked_mut(bytes) }
    }
}
