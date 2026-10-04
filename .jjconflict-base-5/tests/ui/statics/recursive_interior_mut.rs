//@ check-pass

use std::cell::Cell;
use std::ptr::NonNull;

struct ChunkFooter {
    prev: Cell<NonNull<ChunkFooter>>,
}

struct EmptyChunkFooter(ChunkFooter);

unsafe impl Sync for EmptyChunkFooter {}

static EMPTY_CHUNK: EmptyChunkFooter = EmptyChunkFooter(ChunkFooter {
    prev: Cell::new(unsafe {
        NonNull::new_unchecked(&EMPTY_CHUNK as *const EmptyChunkFooter as *mut ChunkFooter)
    }),
});

fn main() {}
