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
}

/// Reserves space for `n` elements in the given vector.
fn reserve[T](&mutable T[] v, uint n) {
    rustrt::ivec_reserve(v, n);
}

fn on_heap[T](&T[] v) -> bool {
    ret rustrt::ivec_on_heap(v);
}

