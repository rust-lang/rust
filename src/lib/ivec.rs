// Interior vector utility functions.

import option::none;
import option::some;
import uint::next_power_of_two;

type operator2[T,U,V] = fn(&T, &U) -> V;

native "rust-intrinsic" mod rusti {
    fn ivec_len[T](&T[] v) -> uint;
}

native "rust" mod rustrt {
    fn ivec_reserve[T](&mutable T[mutable?] v, uint n);
    fn ivec_on_heap[T](&T[] v) -> bool;
    fn ivec_to_ptr[T](&T[] v) -> *T;
    fn ivec_copy_from_buf[T](&mutable T[mutable?] v, *T ptr, uint count);
}

/// Reserves space for `n` elements in the given vector.
fn reserve[T](&mutable T[mutable?] v, uint n) {
    rustrt::ivec_reserve(v, n);
}

fn on_heap[T](&T[] v) -> bool {
    ret rustrt::ivec_on_heap(v);
}

fn to_ptr[T](&T[] v) -> *T {
    ret rustrt::ivec_to_ptr(v);
}

fn len[T](&T[mutable?] v) -> uint {
    ret rusti::ivec_len(v);
}

type init_op[T] = fn(uint) -> T;

fn init_fn[T](&init_op[T] op, uint n_elts) -> T[] {
    auto v = ~[];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[op(i)]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_fn_mut[T](&init_op[T] op, uint n_elts) -> T[mutable] {
    auto v = ~[mutable];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[mutable op(i)]; i += 1u; }
    ret v;
}

fn init_elt[T](&T t, uint n_elts) -> T[] {
    auto v = ~[];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[t]; i += 1u; }
    ret v;
}

// TODO: Remove me once we have slots.
fn init_elt_mut[T](&T t, uint n_elts) -> T[mutable] {
    auto v = ~[mutable];
    reserve(v, n_elts);
    let uint i = 0u;
    while (i < n_elts) { v += ~[mutable t]; i += 1u; }
    ret v;
}


// Accessors

/// Returns the last element of `v`.
fn last[T](&T[mutable?] v) -> option::t[T] {
    if (len(v) == 0u) { ret none; }
    ret some(v.(len(v) - 1u));
}

/// Returns a copy of the elements from [`start`..`end`) from `v`.
fn slice[T](&T[mutable?] v, uint start, uint end) -> T[] {
    assert (start <= end);
    assert (end <= len(v));
    auto result = ~[];
    reserve(result, end - start);
    auto i = start;
    while (i < end) { result += ~[v.(i)]; i += 1u; }
    ret result;
}

// TODO: Remove me once we have slots.
fn slice_mut[T](&T[mutable?] v, uint start, uint end) -> T[mutable] {
    assert (start <= end);
    assert (end <= len(v));
    auto result = ~[mutable];
    reserve(result, end - start);
    auto i = start;
    while (i < end) { result += ~[mutable v.(i)]; i += 1u; }
    ret result;
}


// Mutators

// TODO


// Appending

/// Expands the given vector in-place by appending `n` copies of `initval`.
fn grow[T](&mutable T[] v, uint n, &T initval) {
    reserve(v, next_power_of_two(len(v) + n));
    let uint i = 0u;
    while (i < n) {
        v += ~[initval];
        i += 1u;
    }
}

// TODO: Remove me once we have slots.
fn grow_mut[T](&mutable T[mutable] v, uint n, &T initval) {
    reserve(v, next_power_of_two(len(v) + n));
    let uint i = 0u;
    while (i < n) {
        v += ~[mutable initval];
        i += 1u;
    }
}

/// Calls `f` `n` times and appends the results of these calls to the given
/// vector.
fn grow_fn[T](&mutable T[] v, uint n, fn(uint)->T init_fn) {
    reserve(v, next_power_of_two(len(v) + n));
    let uint i = 0u;
    while (i < n) {
        v += ~[init_fn(i)];
        i += 1u;
    }
}

/// Sets the element at position `index` to `val`. If `index` is past the end
/// of the vector, expands the vector by replicating `initval` to fill the
/// intervening space.
fn grow_set[T](&mutable T[mutable] v, uint index, &T initval, &T val) {
    if (index >= len(v)) { grow_mut(v, index - len(v) + 1u, initval); }
    v.(index) = val;
}

mod unsafe {
    fn copy_from_buf[T](&mutable T[] v, *T ptr, uint count) {
        ret rustrt::ivec_copy_from_buf(v, ptr, count);
    }
}

