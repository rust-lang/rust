// Interior vector utility functions.

import option::none;
import option::some;

type operator2[T,U,V] = fn(&T, &U) -> V;

native "rust-intrinsic" mod rusti {
    fn ivec_len[T](&T[] v) -> uint;
}

native "rust" mod rustrt {
    fn ivec_reserve[T](&mutable T[] v, uint n);
    fn ivec_on_heap[T](&T[] v) -> bool;
    fn ivec_to_ptr[T](&T[] v) -> *T;
    fn ivec_copy_from_buf[T](&mutable T[] v, *T ptr, uint count);
}

/// Reserves space for `n` elements in the given vector.
fn reserve[T](&mutable T[] v, uint n) {
    rustrt::ivec_reserve(v, n);
}

fn on_heap[T](&T[] v) -> bool {
    ret rustrt::ivec_on_heap(v);
}

fn to_ptr[T](&T[] v) -> *T {
    ret rustrt::ivec_to_ptr(v);
}

fn len[T](&T[] v) -> uint {
    ret rusti::ivec_len(v);
}

mod unsafe {
    fn copy_from_buf[T](&mutable T[] v, *T ptr, uint count) {
        ret rustrt::ivec_copy_from_buf(v, ptr, count);
    }
}

