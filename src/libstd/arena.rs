// Dynamic arenas.

export arena, arena_with_size;

import list;
import list::{list, cons, nil};

type chunk = {data: ~[u8], mut fill: uint};
type arena = {mut chunks: @list<@chunk>};

fn chunk(size: uint) -> @chunk {
    let mut v = ~[];
    vec::reserve(v, size);
    @{ data: v, mut fill: 0u }
}

fn arena_with_size(initial_size: uint) -> arena {
    ret {mut chunks: @cons(chunk(initial_size), @nil)};
}

fn arena() -> arena {
    arena_with_size(32u)
}

impl arena for arena {
    fn alloc_grow(n_bytes: uint, align: uint) -> *() {
        // Allocate a new chunk.
        let mut head = list::head(self.chunks);
        let chunk_size = vec::capacity(head.data);
        let new_min_chunk_size = uint::max(n_bytes, chunk_size);
        head = chunk(uint::next_power_of_two(new_min_chunk_size + 1u));
        self.chunks = @cons(head, self.chunks);

        ret self.alloc_inner(n_bytes, align);
    }

    #[inline(always)]
    fn alloc_inner(n_bytes: uint, align: uint) -> *() {
        let alignm1 = align - 1u;
        let mut head = list::head(self.chunks);

        let mut start = head.fill;
        start = (start + alignm1) & !alignm1;
        let end = start + n_bytes;
        if end > vec::capacity(head.data) {
            ret self.alloc_grow(n_bytes, align);
        }

        unsafe {
            let p = ptr::offset(vec::unsafe::to_ptr(head.data), start);
            head.fill = end;
            ret unsafe::reinterpret_cast(p);
        }
    }

    #[inline(always)]
    fn alloc(tydesc: *()) -> *() {
        unsafe {
            let tydesc = tydesc as *sys::type_desc;
            self.alloc_inner((*tydesc).size, (*tydesc).align)
        }
    }
}

