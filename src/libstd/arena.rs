// Dynamic arenas.

export arena, arena_with_size;

import list;

type chunk = {data: [u8], mut fill: uint};
type arena = {mut chunks: list::list<@chunk>};

fn chunk(size: uint) -> @chunk {
    @{ data: vec::from_elem(size, 0u8), mut fill: 0u }
}

fn arena_with_size(initial_size: uint) -> arena {
    ret {mut chunks: list::cons(chunk(initial_size), @list::nil)};
}

fn arena() -> arena {
    arena_with_size(32u)
}

impl arena for arena {
    unsafe fn alloc<T>(n_bytes: uint) -> &self.T {
        let mut head = list::head(self.chunks);
        if head.fill + n_bytes > vec::len(head.data) {
            // Allocate a new chunk.
            let new_min_chunk_size = uint::max(n_bytes, vec::len(head.data));
            head = chunk(uint::next_power_of_two(new_min_chunk_size));
            self.chunks = list::cons(head, @self.chunks);
        }

        let start = vec::unsafe::to_ptr(head.data);
        let p = ptr::offset(start, head.fill);
        head.fill += n_bytes;
        ret unsafe::reinterpret_cast(p);
    }
}

